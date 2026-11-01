"""Action masks for P9 result ordering and pagination."""

from __future__ import annotations

try:
    from . import tool
except ImportError:
    tool = None


def build_mask(action_id: str, state: dict, action_space: list[str]) -> list[str]:
    if tool is None:
        return list(action_space)

    if action_id == "P9A1":
        if state.get("flags", {}).get("P8", {}).get("has_into"):
            allowed = {"no_order"}
        elif tool.is_latest_query(state):
            allowed = {"no_order", "latest_metadata_order"}
        elif tool.result_alignment(state) == "device_alignment":
            allowed = {
                "no_order",
                "time_order",
                "device_order",
                "device_time_order",
                "time_device_order",
                "expression_order",
            }
        else:
            allowed = {"no_order", "time_order", "expression_order"}
        if not tool.common_sortable_expressions(state):
            allowed.discard("expression_order")
        return [action for action in action_space if action in allowed]

    if action_id == "P9A3":
        if tool.is_latest_query(state):
            return [action for action in action_space if action == "no_pagination"]
        masked = list(action_space)
        if not tool.column_pagination_supported(state):
            masked = [
                action
                for action in masked
                if action not in tool.COLUMN_PAGINATION_POLICIES
            ]
        return masked

    return list(action_space)
