"""Action masks for the P3 Group Semantics skill package."""

from __future__ import annotations


GROUP_AGG_DISABLED_MASK = {"grouping_only", "hybrid_grouping"}


def build_mask(action_id: str, state: dict, action_space: list[str]) -> list[str]:
    """Return currently valid candidates for a P3 action-level skill."""

    normalized_action_space = _normalize_action_space(action_space)
    if action_id == "P3A1" and _flag_is_true(state, "latest_time"):
        return ["none"] if "none" in normalized_action_space else normalized_action_space
    if action_id == "P3A2" and _should_force_time_interval(state):
        return (
            ["time_interval"]
            if "time_interval" in normalized_action_space
            else normalized_action_space
        )
    if action_id == "P3A3":
        forced_action = _forced_group_axis_action(
            state=state,
            action_space=normalized_action_space,
        )
        return [forced_action] if forced_action else normalized_action_space
    if action_id != "P3A1":
        return normalized_action_space

    disabled_actions = set()
    tag_disabled = _flag_is_false(
        state=state,
        flag_name="tag_agg",
    )
    if _flag_is_false(state=state, flag_name="group_agg") or (
        _flag_is_false(state=state, flag_name="level_agg")
        and tag_disabled
    ):
        disabled_actions.update(GROUP_AGG_DISABLED_MASK)
    if _flag_is_false(state=state, flag_name="time_group"):
        disabled_actions.add("hybrid_grouping")

    if disabled_actions:
        return [
            action
            for action in normalized_action_space
            if action not in disabled_actions
        ]

    return normalized_action_space


def _normalize_action_space(action_space: list[str]) -> list[str]:
    normalized = []
    seen = set()
    for action in action_space:
        if not action:
            continue
        value = str(action).strip()
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


def _flag_is_false(state: dict, flag_name: str) -> bool:
    """Return True only when a flag is explicitly set to False."""

    if not isinstance(state, dict):
        return False

    flags = state.get("flags", {})
    if not isinstance(flags, dict):
        return False

    for phase_flags in flags.values():
        if isinstance(phase_flags, dict) and flag_name in phase_flags:
            return phase_flags[flag_name] is False

    return flags.get(flag_name) is False


def _flag_is_true(state: dict, flag_name: str) -> bool:
    """Return True only when a flag is explicitly set to True."""

    if not isinstance(state, dict):
        return False

    flags = state.get("flags", {})
    if not isinstance(flags, dict):
        return False

    for phase_flags in flags.values():
        if isinstance(phase_flags, dict) and flag_name in phase_flags:
            return phase_flags[flag_name] is True

    return flags.get(flag_name) is True


def _spo_equals(state: dict, phase_id: str, key: str, value: str) -> bool:
    if not isinstance(state, dict):
        return False

    spo = state.get("spo", {})
    if not isinstance(spo, dict):
        return False

    phase_spo = spo.get(phase_id, {})
    return isinstance(phase_spo, dict) and phase_spo.get(key) == value


def _forced_group_axis_action(state: dict, action_space: list[str]) -> str | None:
    tag_disabled = _flag_is_false(
        state=state,
        flag_name="tag_agg",
    )
    level_disabled = _flag_is_false(state=state, flag_name="level_agg")

    if tag_disabled and not level_disabled and "level" in action_space:
        return "level"
    if level_disabled and not tag_disabled and "tags" in action_space:
        return "tags"
    return None


def _should_force_time_interval(state: dict) -> bool:
    return _flag_is_true(state, "time_group") or _spo_equals(
        state=state,
        phase_id="P3",
        key="group_paradigm",
        value="hybrid_grouping",
    )
