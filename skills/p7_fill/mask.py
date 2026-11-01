"""Action masks for P7 Fill Semantics."""

from __future__ import annotations

try:
    from . import tool
except ImportError:
    tool = None


def build_mask(action_id: str, state: dict, action_space: list[str]) -> list[str]:
    """Return currently valid P7 candidates."""

    if action_id != "P7A1" or tool is None:
        return list(action_space)

    supported_modes = set(tool.supported_fill_modes_for_state(state))
    return [action for action in action_space if action in supported_modes]
