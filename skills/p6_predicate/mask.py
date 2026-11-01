"""Action masks for P6 Predicate Semantics."""

from __future__ import annotations

from typing import Any


def build_mask(action_id: str, state: dict, action_space: list[str]) -> list[str]:
    """Return currently valid P6 candidates."""

    if action_id != "P6A1":
        return list(action_space)

    masked_actions = list(action_space)
    where_allowed = bool(_flag(state, "P2", "where_time")) or bool(
        _flag(state, "P3", "where_value")
    )
    having_allowed = bool(_flag(state, "P3", "having_flag")) and bool(
        _flag(state, "P4", "having_flag2")
    )

    if not where_allowed:
        masked_actions = [
            action
            for action in masked_actions
            if action not in {"where_only", "both"}
        ]
    if not having_allowed:
        masked_actions = [
            action
            for action in masked_actions
            if action not in {"having_only", "both"}
        ]

    return masked_actions


def _flag(state: dict[str, Any], phase_id: str, flag_name: str) -> Any:
    flags = state.get("flags", {})
    if not isinstance(flags, dict):
        return None

    phase_flags = flags.get(phase_id, {})
    if isinstance(phase_flags, dict) and flag_name in phase_flags:
        return phase_flags[flag_name]

    if flag_name in flags:
        return flags[flag_name]

    return None
