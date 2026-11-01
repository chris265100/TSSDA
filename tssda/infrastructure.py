"""Classify external infrastructure failures that must not affect learning."""

from __future__ import annotations

import re
from typing import Any, Iterable


_INFRASTRUCTURE_ERROR_TYPES = {
    "apiconnectionerror",
    "apitimeouterror",
    "connecterror",
    "connecttimeout",
    "connectionerror",
    "internalservererror",
    "iotdbconnectionexception",
    "networkerror",
    "pooltimeout",
    "ratelimiterror",
    "readtimeout",
    "serviceunavailableerror",
    "timeouterror",
    "ttransportexception",
    "writetimeout",
}
_INFRASTRUCTURE_MESSAGE_MARKERS = (
    "insufficient balance",
    "insufficient quota",
    "quota exceeded",
    "rate limit",
    "too many requests",
    "connection aborted",
    "connection refused",
    "connection reset",
    "network error",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
)
_INSUFFICIENT_BALANCE_MARKERS = (
    "insufficient balance",
    "insufficient quota",
    "quota exceeded",
)
_RETRYABLE_HTTP_STATUS = re.compile(
    r"(?:error|status)(?:\s+code)?\s*[:=]?\s*(402|408|425|429|5\d\d)\b",
    re.IGNORECASE,
)


class InsufficientBalanceError(RuntimeError):
    """Raised when a model provider reports exhausted paid capacity."""


def insufficient_balance_error_reason(error: Any) -> str | None:
    """Return a terminal billing/quota reason for an exception or error dict."""

    if isinstance(error, dict):
        error_type = str(error.get("error_type") or error.get("type") or "")
        message = str(
            error.get("error_message")
            or error.get("message")
            or error.get("reason")
            or ""
        )
        status_code = error.get("status_code")
    else:
        error_type = type(error).__name__
        message = str(error)
        status_code = getattr(error, "status_code", None)
        if status_code is None:
            response = getattr(error, "response", None)
            status_code = getattr(response, "status_code", None)

    normalized_message = message.lower()
    if status_code == 402 or any(
        marker in normalized_message
        for marker in _INSUFFICIENT_BALANCE_MARKERS
    ):
        return f"{error_type}: {message}".strip(": ")
    return None


def infrastructure_error_reason(error: dict[str, Any]) -> str | None:
    """Return a concise reason when one error is infrastructure-related."""

    if balance_reason := insufficient_balance_error_reason(error):
        return balance_reason
    error_type = str(error.get("error_type") or error.get("type") or "")
    message = str(
        error.get("error_message")
        or error.get("message")
        or error.get("reason")
        or ""
    )
    normalized_type = error_type.rsplit(".", 1)[-1].lower()
    normalized_message = message.lower()
    if normalized_type in _INFRASTRUCTURE_ERROR_TYPES:
        return f"{error_type}: {message}".strip(": ")
    if any(
        marker in normalized_message
        for marker in _INFRASTRUCTURE_MESSAGE_MARKERS
    ):
        return f"{error_type}: {message}".strip(": ")
    if _RETRYABLE_HTTP_STATUS.search(message):
        return f"{error_type}: {message}".strip(": ")
    return None


def iter_sql_validation_errors(
    validation: dict[str, Any],
) -> Iterable[dict[str, Any]]:
    """Yield generation and failed execution results in validation order."""

    for attempt in validation.get("attempts", []) or []:
        if not isinstance(attempt, dict):
            continue
        generation_error = attempt.get("generation_error")
        if isinstance(generation_error, dict):
            yield generation_error
        for sample in attempt.get("samples", []) or []:
            if isinstance(sample, dict) and not sample.get("passed"):
                yield sample


def state_infrastructure_failure_reason(state: Any) -> str | None:
    """Classify a trajectory only when it produced no storable sample."""

    validation = getattr(state, "sql_validation", {})
    if isinstance(validation, dict):
        accepted_count = validation.get("accepted_count")
        if not isinstance(accepted_count, int):
            accepted_count = len(validation.get("accepted_samples") or []) + len(
                validation.get("unsupported_samples") or []
            )
        if validation.get("passed") and accepted_count > 0:
            return None
        for error in iter_sql_validation_errors(validation):
            if reason := infrastructure_error_reason(error):
                return reason

    if getattr(state, "final_samples", None):
        return None
    for failure in getattr(state, "skill_failures", []) or []:
        if isinstance(failure, dict):
            if reason := infrastructure_error_reason(failure):
                return reason
    for phase in getattr(state, "phase_outputs", []) or []:
        if not isinstance(phase, dict):
            continue
        errors = phase.get("errors") or []
        if isinstance(errors, dict):
            errors = [errors]
        for error in errors:
            if isinstance(error, dict):
                if reason := infrastructure_error_reason(error):
                    return reason
        for skill in phase.get("skill_outputs", []) or []:
            if not isinstance(skill, dict):
                continue
            pipeline_error = skill.get("pipeline_error")
            if isinstance(pipeline_error, dict):
                if reason := infrastructure_error_reason(pipeline_error):
                    return reason
    return None
