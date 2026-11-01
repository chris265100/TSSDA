"""Utility helpers for P7 Fill Semantics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


FILL_MODE_SUPPORT_BY_TYPE = {
    "BOOLEAN": {"previous", "constant"},
    "INT32": {"previous", "linear", "constant"},
    "INT64": {"previous", "linear", "constant"},
    "FLOAT": {"previous", "linear", "constant"},
    "DOUBLE": {"previous", "linear", "constant"},
    "TEXT": {"previous", "constant"},
    "STRING": {"previous", "constant"},
}

ALL_FILL_MODES = {"none", "previous", "linear", "constant"}
NUMERIC_TYPES = {"INT32", "INT64", "FLOAT", "DOUBLE"}


def fill_mode_support_table() -> dict[str, list[str]]:
    """Return a JSON-serializable data type -> fill mode support table."""

    mode_order = ("previous", "linear", "constant")
    return {
        data_type: [mode for mode in mode_order if mode in supported_modes]
        for data_type, supported_modes in FILL_MODE_SUPPORT_BY_TYPE.items()
    }


def temporal_mode_disables_fill(state: dict[str, Any]) -> bool:
    """Return whether earlier temporal semantics should force FILL(none)."""

    temporal_mode = (
        state.get("spo", {})
        .get("P2", {})
        .get("temporal_mode")
    )
    if temporal_mode in {"point_time", "latest_time"}:
        return True

    p2_flags = state.get("flags", {}).get("P2", {})
    return isinstance(p2_flags, dict) and p2_flags.get("latest_time") is True


def supported_fill_modes_for_state(state: dict[str, Any]) -> list[str]:
    """Return fill modes supported by all selected P5 expression data types."""

    if temporal_mode_disables_fill(state):
        return ["none"]

    selected_expressions = selected_select_expressions(state)
    data_types = selected_expression_data_types(state)
    if selected_expressions and len(data_types) < len(selected_expressions):
        return ["none", "previous", "constant"]
    supported = {"previous", "linear", "constant"}
    for data_type in data_types:
        type_support = FILL_MODE_SUPPORT_BY_TYPE.get(data_type)
        if type_support is None:
            supported &= {"previous", "constant"}
            continue
        supported &= type_support

    return ["none", *[mode for mode in ("previous", "linear", "constant") if mode in supported]]


def selected_expression_type_summary(state: dict[str, Any]) -> dict[str, Any]:
    """Return selected P5 expressions and their known data types."""

    selected_expressions = selected_select_expressions(state)
    expression_types = _expression_type_index(state)
    resolved = []
    unresolved = []
    data_types = []

    for expression in selected_expressions:
        normalized_expression = normalize_select_expression(expression)
        descriptor = expression_types.get(normalized_expression)
        if descriptor is None:
            unresolved.append(normalized_expression)
            continue
        data_type = _normalize_data_type(descriptor.get("data_type"))
        resolved.append(
            {
                "expression": normalized_expression,
                "data_type": data_type,
                "source": descriptor.get("source"),
            }
        )
        if data_type and data_type not in data_types:
            data_types.append(data_type)

    return {
        "selected_expressions": selected_expressions,
        "resolved_expression_types": resolved,
        "unresolved_expressions": unresolved,
        "data_types": data_types,
        "supported_fill_modes": supported_fill_modes_for_state(state),
    }


def selected_expression_data_types(state: dict[str, Any]) -> list[str]:
    """Return known data types for expressions currently selected by P5."""

    expression_types = _expression_type_index(state)
    data_types = []
    for expression in selected_select_expressions(state):
        descriptor = expression_types.get(normalize_select_expression(expression))
        if descriptor is None:
            continue
        data_type = _normalize_data_type(descriptor.get("data_type"))
        if data_type and data_type not in data_types:
            data_types.append(data_type)
    return data_types


def selected_select_expressions(state: dict[str, Any]) -> list[str]:
    """Return every expression from the current flat query_frame.select."""

    query_frame = state.get("query_frame_flat", {})
    select_value = query_frame.get("select") if isinstance(query_frame, dict) else None
    groups = []
    if isinstance(select_value, str):
        if select_value.strip():
            groups = [select_value]
    elif isinstance(select_value, list):
        groups = [str(group) for group in select_value if str(group).strip()]

    expressions = []
    seen = set()
    for group in groups:
        for expression in split_top_level_expressions(group):
            normalized = normalize_select_expression(expression)
            if normalized and normalized not in seen:
                expressions.append(normalized)
                seen.add(normalized)
    return expressions


def normalize_select_expression(expression: Any) -> str:
    """Remove aliases and surrounding whitespace from one SELECT expression."""

    text = str(expression or "").strip()
    if not text:
        return ""
    alias_index = _find_top_level_as(text)
    if alias_index is not None:
        text = text[:alias_index].strip()
    return text


def split_top_level_expressions(value: str) -> list[str]:
    """Split a comma-separated expression list without splitting function args."""

    expressions = []
    start = 0
    depth = 0
    quote = None
    for index, char in enumerate(value):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            item = value[start:index].strip()
            if item:
                expressions.append(item)
            start = index + 1
    item = value[start:].strip()
    if item:
        expressions.append(item)
    return expressions


def _expression_type_index(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    operator_store = state.get("operator_store", {})
    if not isinstance(operator_store, dict):
        return {}

    index = {}
    for source in ("AggExpr", "SeriesExpr"):
        expressions = operator_store.get(source, [])
        if not isinstance(expressions, list):
            continue
        for descriptor in expressions:
            if not isinstance(descriptor, dict):
                continue
            descriptor_copy = deepcopy(descriptor)
            descriptor_copy["source"] = source
            for key in ("name", "expression"):
                value = normalize_select_expression(descriptor.get(key))
                if value:
                    index[value] = descriptor_copy
    return index


def _normalize_data_type(value: Any) -> str | None:
    if value is None:
        return None
    data_type = str(value).strip().upper()
    if not data_type:
        return None
    if data_type == "BOOLEAN":
        return "BOOLEAN"
    if data_type in {"TEXT", "STRING"}:
        return "TEXT"
    if data_type in NUMERIC_TYPES:
        return data_type
    return data_type


def _find_top_level_as(expression: str) -> int | None:
    lowered = expression.lower()
    depth = 0
    quote = None
    index = 0
    alias_index = None
    while index < len(expression):
        char = expression[index]
        if quote:
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and lowered.startswith(" as ", index):
            alias_index = index
        index += 1
    return alias_index
