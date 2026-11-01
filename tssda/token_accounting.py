"""Dependency-free token accounting for Memory effectiveness metrics."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable


ESTIMATOR_NAME = "mixed_text_estimator_v1"
_TOKEN_PARTS = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|"
    r"[A-Za-z]+|"
    r"\d+(?:\.\d+)?|"
    r"[^\s]"
)


def estimate_text_tokens(value: str) -> int:
    """Estimate tokens consistently without downloading a model tokenizer.

    CJK characters and punctuation count individually. ASCII words and numbers
    use a four-character approximation. This is intentionally reported as an
    estimate; provider-reported prompt tokens are recorded separately whenever
    a real model request occurs.
    """

    total = 0
    for part in _TOKEN_PARTS.findall(str(value or "")):
        if len(part) == 1 and ord(part) >= 0x3400:
            total += 1
        elif part[0].isalnum() and all(char.isascii() for char in part):
            total += max(1, math.ceil(len(part) / 4))
        else:
            total += 1
    return total


def estimate_message_tokens(messages: Iterable[dict[str, Any]]) -> int:
    """Estimate chat input tokens, including small per-message overheads."""

    total = 2
    for message in messages:
        total += 4
        total += estimate_text_tokens(str(message.get("role") or ""))
        total += estimate_text_tokens(str(message.get("content") or ""))
        if message.get("name"):
            total += estimate_text_tokens(str(message["name"]))
    return total


def estimate_json_tokens(value: Any) -> int:
    """Estimate tokens for a JSON payload used by a fallback test delegate."""

    return estimate_text_tokens(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    )


def percentage(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return round(100.0 * float(numerator) / float(denominator), 6)


def empty_memory_token_summary() -> dict[str, Any]:
    return {
        "estimator": ESTIMATOR_NAME,
        "memory_lookup_attempts": 0,
        "memory_hit_attempts": 0,
        "memory_miss_attempts": 0,
        "estimated_saved_input_tokens": 0,
        "estimated_actual_input_tokens": 0,
        "reported_actual_input_tokens": 0,
        "reported_actual_input_token_attempts": 0,
        "estimated_memory_token_efficiency_percent": None,
        "reported_prompt_token_coverage_percent": None,
        "per_skill": {},
    }


def summarize_memory_token_events(
    events: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate retry-aware Memory token events for one trajectory."""

    summary = empty_memory_token_summary()
    per_skill: dict[str, dict[str, Any]] = {}
    for raw_event in events:
        event = dict(raw_event)
        hit = bool(event.get("hit"))
        estimated = max(0, int(event.get("estimated_input_tokens") or 0))
        reported = event.get("reported_actual_input_tokens")
        if isinstance(reported, bool) or not isinstance(reported, int):
            reported = None

        summary["memory_lookup_attempts"] += 1
        if hit:
            summary["memory_hit_attempts"] += 1
            summary["estimated_saved_input_tokens"] += estimated
        else:
            summary["memory_miss_attempts"] += 1
            summary["estimated_actual_input_tokens"] += estimated
            if reported is not None:
                summary["reported_actual_input_tokens"] += max(0, reported)
                summary["reported_actual_input_token_attempts"] += 1

        skill_id = str(event.get("skill_id") or "unknown")
        bucket = per_skill.setdefault(
            skill_id,
            {
                "phase_id": event.get("phase_id"),
                "action_id": event.get("action_id"),
                "lookup_attempts": 0,
                "hits": 0,
                "misses": 0,
                "estimated_saved_input_tokens": 0,
                "estimated_actual_input_tokens": 0,
                "reported_actual_input_tokens": 0,
                "reported_actual_input_token_attempts": 0,
            },
        )
        bucket["lookup_attempts"] += 1
        if hit:
            bucket["hits"] += 1
            bucket["estimated_saved_input_tokens"] += estimated
        else:
            bucket["misses"] += 1
            bucket["estimated_actual_input_tokens"] += estimated
            if reported is not None:
                bucket["reported_actual_input_tokens"] += max(0, reported)
                bucket["reported_actual_input_token_attempts"] += 1

    estimated_denominator = (
        summary["estimated_saved_input_tokens"]
        + summary["estimated_actual_input_tokens"]
    )
    summary["estimated_memory_token_efficiency_percent"] = percentage(
        summary["estimated_saved_input_tokens"],
        estimated_denominator,
    )
    summary["reported_prompt_token_coverage_percent"] = percentage(
        summary["reported_actual_input_token_attempts"],
        summary["memory_miss_attempts"],
    )
    for bucket in per_skill.values():
        bucket["hit_rate_percent"] = percentage(
            bucket["hits"], bucket["lookup_attempts"]
        )
        denominator = (
            bucket["estimated_saved_input_tokens"]
            + bucket["estimated_actual_input_tokens"]
        )
        bucket["estimated_memory_token_efficiency_percent"] = percentage(
            bucket["estimated_saved_input_tokens"], denominator
        )
    summary["per_skill"] = per_skill
    return summary
