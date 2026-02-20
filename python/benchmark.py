from __future__ import annotations

from dataclasses import dataclass
from typing import Any


BENCHMARK_CASES: list[dict[str, Any]] = [
    {
        "name": "lights_simple",
        "dialogue": ["Turn on the kitchen lights."],
        "expected": {"name": "toggle_lights", "arguments": {"room": "kitchen", "state": "on"}},
    },
    {
        "name": "thermostat_basic",
        "dialogue": ["Set thermostat to 70 degrees."],
        "expected": {"name": "set_thermostat", "arguments": {"temperature": 70}},
    },
    {
        "name": "door_lock",
        "dialogue": ["Lock the garage door."],
        "expected": {"name": "lock_door", "arguments": {"door": "garage", "state": "lock"}},
    },
    {
        "name": "scene",
        "dialogue": ["Activate bedtime scene."],
        "expected": {"name": "set_scene", "arguments": {"scene": "bedtime"}},
    },
    {
        "name": "status",
        "dialogue": ["What is the thermostat status?"],
        "expected": {"name": "get_device_status", "arguments": {"device_type": "thermostat"}},
    },
    {
        "name": "contextual_followup",
        "dialogue": ["Turn on lights in the office.", "Now switch them off."],
        "expected": {"name": "toggle_lights", "arguments": {"room": "office", "state": "off"}},
    },
]


@dataclass
class BenchmarkResult:
    model: str
    accuracy: float
    correct: int
    total: int
    avg_latency_ms: float


def _matches_expected(actual: dict | str | None, expected: dict[str, Any]) -> bool:
    if not isinstance(actual, dict):
        return False
    if actual.get("name") != expected.get("name"):
        return False

    actual_args = actual.get("arguments", {})
    expected_args = expected.get("arguments", {})
    if not isinstance(actual_args, dict):
        return False

    for key, value in expected_args.items():
        if actual_args.get(key) != value:
            return False

    return True


def run_benchmark(
    model_name: str,
    port: int,
    api_key: str,
    debug: bool,
    slm_client_cls: Any,
    orchestrator_cls: Any,
    benchmark_cases: list[dict[str, Any]] | None = None,
) -> BenchmarkResult:
    slm = slm_client_cls(model_name=model_name, api_key=api_key, port=port)
    orchestrator = orchestrator_cls(slm, debug=debug)

    cases = benchmark_cases if benchmark_cases is not None else BENCHMARK_CASES
    total_cases = len(cases)
    correct_cases = 0
    latencies: list[float] = []

    for case in cases:
        orchestrator.reset()
        for utterance in case["dialogue"]:
            _ = orchestrator.process_utterance(utterance)
            latencies.append(orchestrator.last_latency_ms)

        if _matches_expected(orchestrator.last_function_call, case["expected"]):
            correct_cases += 1

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    accuracy = correct_cases / total_cases if total_cases else 0.0

    return BenchmarkResult(
        model=model_name,
        accuracy=accuracy,
        correct=correct_cases,
        total=total_cases,
        avg_latency_ms=avg_latency,
    )


def print_benchmark_report(results: list[BenchmarkResult]) -> None:
    print("\nBenchmark results (tool-calling):")
    print("-" * 84)
    print(f"{'Model':40} {'Accuracy':>12} {'Correct':>10} {'Avg Latency':>18}")
    print("-" * 84)
    for res in results:
        print(
            f"{res.model:40} "
            f"{res.accuracy * 100:10.1f}% "
            f"{res.correct:>5}/{res.total:<4} "
            f"{res.avg_latency_ms:14.1f} ms"
        )
    print("-" * 84)
