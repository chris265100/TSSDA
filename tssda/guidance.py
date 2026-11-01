"""Global user guidance shared by semantic and SQL-generation agents."""

from __future__ import annotations

DEFAULT_USER_GENERATION_GOAL = ""


def user_goal_from_input_request(input_request: object) -> str:
    """Extract the optional free-text goal from a trajectory request."""

    request = input_request if isinstance(input_request, dict) else {}
    return str(request.get("user_generation_goal") or "").strip()
