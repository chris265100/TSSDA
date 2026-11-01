"""Runtime helpers for P9 ordering and pagination semantics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

try:
    from skills.p8_into import tool as p8_tool
except ImportError:
    p8_tool = None


COLUMN_PAGINATION_POLICIES = {
    "column_limit",
    "column_limit_offset",
    "row_column_limit",
    "row_column_limit_offset",
}


def result_alignment(state: dict[str, Any]) -> str:
    p8_alignment = state.get("spo", {}).get("P8", {}).get("result_alignment")
    if p8_alignment in {"time_alignment", "device_alignment"}:
        return p8_alignment
    return (
        "device_alignment"
        if state.get("query_frame_flat", {}).get("align_by") == "device"
        else "time_alignment"
    )


def is_latest_query(state: dict[str, Any]) -> bool:
    return bool(
        state.get("flags", {}).get("P2", {}).get("latest_time")
        or state.get("spo", {}).get("P2", {}).get("temporal_mode")
        == "latest_time"
        or state.get("query_frame_flat", {}).get("sql_query_mode")
        == "select_last"
        or state.get("query_frame_flat", {}).get("query_type") == "select_last"
    )


def common_sortable_expressions(state: dict[str, Any]) -> list[str]:
    """Return SELECT expressions available in every P5 expression group."""

    groups = _select_expression_groups(state)
    expression_groups = [
        [_strip_alias(expression) for expression in _split_expressions(group)]
        for group in groups
    ]
    if not expression_groups:
        return []
    common = set(expression_groups[0])
    for expressions in expression_groups[1:]:
        common.intersection_update(expressions)
    return [expression for expression in expression_groups[0] if expression in common]


def result_shape_summary(state: dict[str, Any]) -> dict[str, Any]:
    """Describe value-column counts relevant to SLIMIT/SOFFSET generation."""

    alignment = result_alignment(state)
    group_shapes: list[dict[str, Any]] = []
    source_device_count = _source_device_count(state)
    if p8_tool is not None:
        try:
            plan = p8_tool.build_writeback_cardinality_plan(state, alignment)
        except (TypeError, ValueError):
            plan = None
        if isinstance(plan, dict):
            source_device_count = int(
                plan.get("source_device_count") or source_device_count
            )
            for group in plan.get("select_groups", []):
                if alignment == "device_alignment":
                    value_column_count = group.get("expression_count")
                else:
                    value_column_count = group.get("target_series_count")
                if isinstance(value_column_count, int) and value_column_count > 0:
                    group_shapes.append(
                        {
                            "select_index": group.get("select_index"),
                            "select": group.get("select"),
                            "value_column_count": value_column_count,
                        }
                    )

    if not group_shapes:
        for index, group in enumerate(_select_expression_groups(state)):
            expression_count = len(_split_expressions(group))
            multiplier = (
                max(source_device_count, 1)
                if alignment == "time_alignment"
                else 1
            )
            group_shapes.append(
                {
                    "select_index": index,
                    "select": group,
                    "value_column_count": expression_count * multiplier,
                }
            )

    counts = [group["value_column_count"] for group in group_shapes]
    return {
        "alignment": alignment,
        "latest_query": is_latest_query(state),
        "has_into": bool(state.get("flags", {}).get("P8", {}).get("has_into")),
        "source_device_count": source_device_count,
        "select_groups": deepcopy(group_shapes),
        "minimum_value_column_count": min(counts) if counts else 0,
        "maximum_value_column_count": max(counts) if counts else 0,
    }


def column_pagination_supported(state: dict[str, Any]) -> bool:
    shape = result_shape_summary(state)
    return bool(
        not shape["has_into"]
        and not shape["latest_query"]
        and shape["minimum_value_column_count"] > 1
    )


def selected_ordering_mode(state: dict[str, Any]) -> str | None:
    return state.get("spo", {}).get("P9", {}).get("ordering_mode")


def selected_pagination_policy(state: dict[str, Any]) -> str | None:
    return state.get("spo", {}).get("P9", {}).get("pagination_policy")


def _select_expression_groups(state: dict[str, Any]) -> list[str]:
    value = state.get("query_frame_flat", {}).get("select")
    if isinstance(value, str):
        groups = [value]
    elif isinstance(value, list):
        groups = value
    else:
        groups = []
    return [str(group).strip() for group in groups if str(group).strip()]


def _split_expressions(group: str) -> list[str]:
    if p8_tool is not None:
        return p8_tool.split_top_level_expressions(group)

    expressions = []
    start = 0
    depth = 0
    for index, char in enumerate(group):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            expressions.append(group[start:index].strip())
            start = index + 1
    expressions.append(group[start:].strip())
    return [expression for expression in expressions if expression]


def _strip_alias(expression: str) -> str:
    lowered = expression.lower()
    depth = 0
    quote: str | None = None
    index = 0
    while index < len(expression):
        char = expression[index]
        if quote is not None:
            if char == quote:
                quote = None
        elif char in {"'", '"', "`"}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and lowered.startswith(" as ", index):
            return expression[:index].strip()
        index += 1
    return expression.strip()


def _source_device_count(state: dict[str, Any]) -> int:
    schema = state.get("schema", {})
    schema_view = schema.get("working_schema") or schema.get("meta_schema") or {}
    count = schema_view.get("device_count")
    if isinstance(count, int) and count > 0:
        return count
    devices = schema_view.get("devices", [])
    return len(devices) if isinstance(devices, list) else 0
