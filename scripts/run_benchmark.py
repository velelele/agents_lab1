"""
Requirement 4.1 — прогоны двух локальных моделей с проверкой качества.
Сохраняет отдельный JSON-лог в формате .log для каждой модели.
"""
import sys
import json
import re
import argparse
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LOCAL_MODELS, LOGS_DIR
from src.providers.llama_provider import LlamaCppProvider
from src.providers.maple_server_provider import MapleServerProvider
from src.agent import ToolAgent
from src.prompt_templates import build_system_prompt


TEST_CASES = [
    {
        "query": "Какая сейчас погода в Москве?",
        "expected_tools": ["get_weather"],
        "answer_patterns": [r"\+14\s*°?C"],
    },
    {
        "query": "Сколько будет 256 * 48 + 100?",
        "expected_tools": ["calculate"],
        "answer_patterns": [r"12388"],
    },
    {
        "query": "Какой текущий курс доллара к рублю?",
        "expected_tools": ["get_exchange_rate"],
        "answer_patterns": [r"92[.,]40"],
    },
    {
        "query": "Который сейчас час?",
        "expected_tools": ["get_current_time"],
        "answer_patterns": [r"12:30"],
    },
    {
        "query": "Какой статус кондиционера?",
        "expected_tools": ["get_device_status"],
        "answer_patterns": [r"21\s*°?C"],
    },
    {
        "query": "Привет! Расскажи кратко, что такое нейронная сеть.",
        "expected_tools": [],
        "answer_patterns": [],
    },
    {
        "query": "Какая погода в Токио и какой курс евро?",
        "expected_tools": ["get_weather", "get_exchange_rate"],
        "answer_patterns": [r"\+22\s*°?C", r"1\s*EUR\s*=\s*100[.,]85\s*RUB"],
    },
    {
        "query": "Посчитай 15% от 8500",
        "expected_tools": ["calculate"],
        "answer_patterns": [r"1275(?:[.,]0+)?"],
    },
]


def evaluate_result(case: dict, result: dict) -> dict:
    """Check required tool calls and evidence in the final answer."""
    called_tools = [step["action"] for step in result["steps"] if step.get("action")]
    seen_calls = set()
    duplicate_calls = []
    for step in result["steps"]:
        action = step.get("action")
        if not action:
            continue
        args = json.dumps(step.get("args") or {}, ensure_ascii=False, sort_keys=True)
        signature = (action, args)
        if signature in seen_calls:
            duplicate_calls.append({"action": action, "args": json.loads(args)})
        else:
            seen_calls.add(signature)
    expected_tools = case["expected_tools"]
    missing_tools = [name for name in expected_tools if name not in called_tools]
    unexpected_tools = [name for name in called_tools if name not in expected_tools]
    answer = result["final_answer"]
    missing_patterns = [
        pattern for pattern in case["answer_patterns"]
        if not re.search(pattern, answer, re.IGNORECASE)
    ]

    issues = []
    if not answer.strip():
        issues.append("Пустой итоговый ответ")
    if missing_tools:
        issues.append("Не вызваны инструменты: " + ", ".join(missing_tools))
    if unexpected_tools:
        issues.append("Лишние вызовы инструментов: " + ", ".join(unexpected_tools))
    if duplicate_calls:
        labels = [f"{item['action']}({json.dumps(item['args'], ensure_ascii=False, sort_keys=True)})" for item in duplicate_calls]
        issues.append("Повторные вызовы инструментов: " + ", ".join(labels))
    if missing_patterns:
        issues.append("В ответе нет ожидаемых результатов: " + ", ".join(missing_patterns))

    return {
        "expected_tools": expected_tools,
        "called_tools": called_tools,
        "missing_tools": missing_tools,
        "unexpected_tools": unexpected_tools,
        "duplicate_calls": duplicate_calls,
        "answer_patterns": case["answer_patterns"],
        "missing_answer_patterns": missing_patterns,
        "quality_issues": issues,
        "quality_passed": not issues,
    }


def run_benchmark(model_key: str, case_indexes=None) -> list:
    """Run all benchmark cases through one local model."""
    info = LOCAL_MODELS[model_key]
    if not info["path"].exists():
        print(f"[SKIP] Модель {model_key} не найдена: {info['path'].name}")
        return []

    print(f"\n{'='*60}")
    print(f"Бенчмарк модели: {info['name']} ({info['path'].name})")
    print(f"{'='*60}")

    if model_key == "maple":
        provider = MapleServerProvider(model_path=info["path"], model_alias=model_key)
    else:
        provider = LlamaCppProvider(
            model_path=info["path"],
            model_alias=model_key,
            chat_format=info["chat_format"],
            verbose=False,
        )
    agent = ToolAgent(provider=provider, verbose=False)
    results = []

    for index, case in enumerate(TEST_CASES, 1):
        if case_indexes is not None and index not in case_indexes:
            continue
        query = case["query"]
        print(f"\n[{index}/{len(TEST_CASES)}] Запрос: {query}")
        try:
            response = agent.run(user_message=query)
            record = {
                "query_index": index,
                "query": query,
                "model": model_key,
                "model_name": info["name"],
                "final_answer": response["final_answer"],
                "steps_count": len(response["steps"]),
                "total_time_sec": round(response["total_time"], 2),
                "steps_detail": [
                    {
                        "step": step["step"],
                        "action": step["action"],
                        "args": step["args"],
                        "observation": step["observation"],
                        "raw_response": step["raw_response"][:500],
                    }
                    for step in response["steps"]
                ],
                "execution_successful": True,
            }
            record.update(evaluate_result(case, response))
            results.append(record)

            marker = "✅" if record["quality_passed"] else "⚠️"
            print(f"  {marker} Ответ: {response['final_answer'][:160]}")
            print(f"  Инструменты: {record['called_tools'] or 'нет'}")
            if record["quality_issues"]:
                print("  Проблемы: " + "; ".join(record["quality_issues"]))
            print(f"  Время: {response['total_time']:.2f}с, шагов: {len(response['steps'])}")
        except Exception as exc:
            results.append(
                {
                    "query_index": index,
                    "query": query,
                    "model": model_key,
                    "execution_successful": False,
                    "quality_passed": False,
                    "quality_issues": [f"Ошибка выполнения: {exc}"],
                    "error": str(exc),
                }
            )
            print(f"  ❌ Ошибка выполнения: {exc}")

    close = getattr(provider, "close", None)
    if close:
        close()
    return results


def save_log(model_key: str, results: list) -> Path:
    """Save the run and its quality checks to a separate JSON .log file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"benchmark_{model_key}_{timestamp}.log"
    execution_count = sum(bool(row.get("execution_successful")) for row in results)
    passed_count = sum(bool(row.get("quality_passed")) for row in results)
    durations = [row["total_time_sec"] for row in results if "total_time_sec" in row]

    log_data = {
        "model": model_key,
        "model_name": LOCAL_MODELS[model_key]["name"],
        "timestamp": timestamp,
        "total_queries": len(results),
        "execution_successful": execution_count,
        "quality_passed": passed_count,
        "successful": passed_count,
        "failed": len(results) - passed_count,
        "avg_time_sec": round(sum(durations) / max(len(durations), 1), 2),
        "quality_definition": "Все ожидаемые инструменты вызваны, лишних вызовов нет, ответ содержит проверяемые результаты.",
        "prompt_version": "maple_prompt_iteration_v12",
        "system_prompt": build_system_prompt(),
        "results": results,
    }

    with log_path.open("w", encoding="utf-8") as log_file:
        json.dump(log_data, log_file, ensure_ascii=False, indent=2)
    print(f"\nЛог сохранен: {log_path}")
    print(f"Качество: {passed_count}/{len(results)}; выполнение без исключений: {execution_count}/{len(results)}")
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Бенчмарк локальных моделей для пунктов 4.1 и 4.2")
    parser.add_argument("--models", nargs="+", choices=sorted(LOCAL_MODELS), default=["qwen", "llama"])
    parser.add_argument("--cases", nargs="+", type=int, choices=range(1, len(TEST_CASES) + 1),
                        help="Ограничить прогон номерами кейсов; по умолчанию запускаются все")
    args = parser.parse_args()

    print("=" * 60)
    print("  Лабораторная работа — проверка tool-вызовов")
    print("  Модели: " + ", ".join(args.models))
    print("=" * 60)

    for model_key in args.models:
        results = run_benchmark(model_key, set(args.cases) if args.cases else None)
        if results:
            save_log(model_key, results)


if __name__ == "__main__":
    main()
