"""Retry and trajectory-control errors for model-driven skill execution."""

from __future__ import annotations

import traceback
from typing import Any


MAX_MODEL_OUTPUT_RETRIES = 2
MAX_FAILED_SKILLS_PER_TRAJECTORY = 3


def build_model_output_error(
    exc: Exception,
    *,
    attempt: int,
) -> dict[str, Any]:
    """Build a serializable diagnostic record for one failed attempt."""

    return {
        "attempt": attempt,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def build_retry_instruction(error: dict[str, Any]) -> str:
    """Create the corrective instruction supplied to the next model attempt."""

    return (
        "上次输出产生了错误，报错如下：\n"
        f"{error.get('error_type', 'Error')}: {error.get('message', '')}\n"
        "请根据报错修正本次输出，严格遵循原始 output_instruction 和输出格式，"
        "不要添加解释、Markdown 或格式之外的字段。"
    )


class TrajectoryFailureLimitExceeded(RuntimeError):
    """Raised after a trajectory accumulates too many failed skills."""

    def __init__(self, *, state: Any, failure: dict[str, Any]):
        self.state = state
        self.failure = failure
        super().__init__(
            "Trajectory aborted after "
            f"{state.failed_skill_count} failed skills; latest failure: "
            f"{failure.get('skill_id')}"
        )
