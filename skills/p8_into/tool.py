"""Runtime helpers for P8 INTO semantics."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


def writeback_requested(state: dict[str, Any]) -> bool:
    """Return whether P0 selected a materialization trajectory."""

    p0_flags = state.get("flags", {}).get("P0", {})
    if "writeback_required" in p0_flags:
        return bool(p0_flags["writeback_required"])
    return (
        state.get("spo", {}).get("P0", {}).get("task_family")
        == "materialization"
    )


def latest_query(state: dict[str, Any]) -> bool:
    """Return whether upstream phases selected SELECT LAST semantics."""

    query_frame = state.get("query_frame_flat", {})
    return bool(
        state.get("flags", {}).get("P2", {}).get("latest_time")
        or state.get("spo", {}).get("P2", {}).get("temporal_mode")
        == "latest_time"
        or state.get("spo", {}).get("P4", {}).get("operator_regime")
        == "latest_query"
        or query_frame.get("sql_query_mode") == "select_last"
        or query_frame.get("query_type") == "select_last"
    )


def into_incompatibility_reasons(state: dict[str, Any]) -> list[str]:
    """Return upstream grammar conflicts that prohibit SELECT INTO."""

    reasons: list[str] = []
    query_frame = state.get("query_frame_flat", {})
    p3_spo = state.get("spo", {}).get("P3", {})

    if latest_query(state):
        reasons.append("LAST queries do not support INTO")

    group_axis = p3_spo.get("group_axis")
    group_by = query_frame.get("group_by") or []
    group_by_text = " ".join(
        str(value) for value in (group_by if isinstance(group_by, list) else [group_by])
    ).upper()
    if group_axis == "tags" or "TAGS(" in group_by_text:
        reasons.append("GROUP BY TAGS does not support INTO")

    if query_frame.get("slimit") is not None:
        reasons.append("SLIMIT does not support INTO")
    if query_frame.get("soffset") is not None:
        reasons.append("SOFFSET does not support INTO")
    return reasons


def writeback_eligible(state: dict[str, Any]) -> bool:
    return not into_incompatibility_reasons(state)


def active_schema_view(state: dict[str, Any]) -> dict[str, Any]:
    """Return working schema when active, otherwise compact meta schema."""

    schema = state.get("schema", {})
    return deepcopy(
        schema.get("working_schema")
        or schema.get("meta_schema")
        or {}
    )


def source_devices(state: dict[str, Any]) -> list[dict[str, Any]]:
    devices = active_schema_view(state).get("devices", [])
    return [deepcopy(device) for device in devices if isinstance(device, dict)]


def supports_device_alignment(state: dict[str, Any]) -> bool:
    """Check whether the working scope is meaningful for ALIGN BY DEVICE."""

    devices = source_devices(state)
    if len(devices) <= 1:
        return False
    return not _has_same_name_type_conflicts(devices)


def select_groups(state: dict[str, Any]) -> list[str]:
    value = state.get("query_frame_flat", {}).get("select")
    if isinstance(value, str):
        groups = [value]
    elif isinstance(value, list):
        groups = value
    else:
        groups = []
    normalized = [str(group).strip() for group in groups if str(group).strip()]
    if not normalized:
        raise ValueError("P8 requires a non-empty query_frame.select")
    return normalized


def split_top_level_expressions(value: str) -> list[str]:
    """Split a SELECT group without splitting function argument commas."""

    expressions: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    index = 0
    while index < len(value):
        char = value[index]
        if quote is not None:
            if char == quote:
                if index + 1 < len(value) and value[index + 1] == quote:
                    index += 1
                else:
                    quote = None
        elif char in {"'", '"', "`"}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("SELECT expression has unbalanced parentheses")
        elif char == "," and depth == 0:
            expression = value[start:index].strip()
            if not expression:
                raise ValueError("SELECT group contains an empty expression")
            expressions.append(expression)
            start = index + 1
        index += 1
    if quote is not None or depth != 0:
        raise ValueError("SELECT expression has unbalanced quotes or parentheses")
    final_expression = value[start:].strip()
    if not final_expression:
        raise ValueError("SELECT group contains an empty expression")
    expressions.append(final_expression)
    return expressions


def build_writeback_cardinality_plan(
    state: dict[str, Any],
    alignment: str,
) -> dict[str, Any]:
    """Build the target cardinality contract consumed by P8A3."""

    if alignment not in {"time_alignment", "device_alignment"}:
        raise ValueError(f"Unsupported P8 alignment mode: {alignment!r}")

    devices = source_devices(state)
    device_paths = [
        str(device.get("path")).strip()
        for device in devices
        if device.get("path")
    ]
    if not device_paths:
        raise ValueError("P8 requires at least one source device")

    group_plans = []
    for select_index, group in enumerate(select_groups(state)):
        expressions = split_top_level_expressions(group)
        expression_count = len(expressions)
        if alignment == "device_alignment":
            target_device_count = len(device_paths)
            measurements_per_target_device = expression_count
            target_series_count = target_device_count * expression_count
        else:
            expansion_counts = [
                _time_aligned_expansion_count(expression, devices)
                for expression in expressions
            ]
            target_series_count = sum(expansion_counts)
            target_device_count = None
            measurements_per_target_device = None

        group_plans.append(
            {
                "select_index": select_index,
                "select": group,
                "expressions": expressions,
                "expression_count": expression_count,
                "time_aligned_expression_expansion": (
                    expansion_counts if alignment == "time_alignment" else None
                ),
                "target_series_count": target_series_count,
                "target_device_count": target_device_count,
                "measurements_per_target_device": measurements_per_target_device,
            }
        )

    return {
        "alignment": alignment,
        "source_device_count": len(device_paths),
        "source_devices": device_paths,
        "select_groups": group_plans,
        "cardinality_rule": (
            "For time alignment, the total target-series count equals the "
            "expanded result value-column count. For device alignment, the "
            "target-device count equals the source-device count and every "
            "target device has one target measurement per SELECT expression."
        ),
    }


def _time_aligned_expansion_count(
    expression: str,
    devices: list[dict[str, Any]],
) -> int:
    """Estimate concrete value columns produced under default time alignment."""

    expression_without_alias = re.split(
        r"\s+AS\s+",
        expression,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    if re.search(r"\broot(?:\.[A-Za-z0-9_`*]+)+", expression_without_alias):
        return 1
    return max(len(devices), 1)


def _has_same_name_type_conflicts(devices: list[dict[str, Any]]) -> bool:
    types_by_name: dict[str, set[str]] = {}
    for device in devices:
        measurements = device.get("measurements", [])
        if not isinstance(measurements, list):
            continue
        for measurement in measurements:
            if not isinstance(measurement, dict) or not measurement.get("name"):
                continue
            name = str(measurement["name"])
            data_type = measurement.get("data_type")
            if data_type:
                types_by_name.setdefault(name, set()).add(str(data_type).upper())
    return any(len(data_types) > 1 for data_types in types_by_name.values())
