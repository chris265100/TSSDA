"""Action masks for P8 result alignment and INTO writeback."""

from __future__ import annotations

try:
    from . import tool
except ImportError:
    tool = None


def build_mask(action_id: str, state: dict, action_space: list[str]) -> list[str]:
    if action_id == "P8A2":
        return [action for action in action_space if action == "no_writeback"]
    if tool is None:
        return list(action_space)

    if action_id == "P8A1":
        if tool.latest_query(state):
            return [
                action
                for action in action_space
                if action == "time_alignment"
            ]
        if tool.supports_device_alignment(state):
            return list(action_space)
        return [action for action in action_space if action != "device_alignment"]

    return list(action_space)
