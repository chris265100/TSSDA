"""Action masks for the P4 Operator Semantics skill package."""

from __future__ import annotations


def build_mask(action_id: str, state: dict, action_space: list[str]) -> list[str]:
    """Return currently valid candidates for a P4 action-level skill."""

    candidates = _normalize_action_space(action_space)
    if action_id == "P4A1":
        if _flag_is_true(state, "P2", "latest_time"):
            return ["latest_query"] if "latest_query" in candidates else candidates
        candidates = [action for action in candidates if action != "latest_query"]
        if _flag_is_true(state, "P3", "group_by_enabled"):
            candidates = [
                action
                for action in candidates
                if action != "timeseries_function_query"
            ]
        return candidates

    measurement_count = _flag_value(state, "P1", "measurement_count")
    if isinstance(measurement_count, int) and measurement_count <= 1:
        if action_id == "P4A2":
            return [
                action for action in candidates if action != "extend_binary_series"
            ]
        if action_id == "P4A3":
            return [
                action for action in candidates if action != "extend_binary_agg_expr"
            ]
    return candidates


def _normalize_action_space(action_space: list[str]) -> list[str]:
    normalized = []
    seen = set()
    for action in action_space:
        value = str(action).strip()
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


def _flag_value(state: dict, phase_id: str, flag_name: str):
    if not isinstance(state, dict):
        return None
    flags = state.get("flags", {})
    if not isinstance(flags, dict):
        return None
    phase_flags = flags.get(phase_id, {})
    if isinstance(phase_flags, dict) and flag_name in phase_flags:
        return phase_flags[flag_name]
    return flags.get(flag_name)


def _flag_is_true(state: dict, phase_id: str, flag_name: str) -> bool:
    return _flag_value(state, phase_id, flag_name) is True
