"""Runtime profile extraction for TSSDA state-level memory."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Iterable

from .temporal import SAMPLING_FIELDS, sampling_features

from .models import (
    ClauseFlag,
    MemoryIndex,
    NONE,
    NOT_APPLICABLE,
    QueryScopeProfile,
    QueryStructureProfile,
    SchemaProfile,
    SelectLayoutProfile,
    StateMemoryProfile,
    TemporalProfile,
    UNKNOWN,
    UNSET,
    WindowShapeProfile,
    cardinality_bucket,
)


NUMERIC_TYPES = {"INT32", "INT64", "FLOAT", "DOUBLE"}
TEXT_TYPES = {"TEXT", "STRING"}
BOOLEAN_TYPES = {"BOOLEAN"}

ISO_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
DURATION_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(ms|mo|[smhdwy])(?!\w)", re.I)
TIME_COMPARISON_RE = re.compile(
    r"\btime\s*(>=|>|<=|<|=)\s*['\"]?"
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}|\d{10,16})",
    re.I,
)


@dataclass
class MemoryRuntimeContext:
    """Trajectory-local inputs needed to build immutable memory snapshots."""

    intent: str | None = None
    intent_pattern: str | None = None
    temporal_observations: dict[str, Any] = field(default_factory=dict)

    def capture_p0_spo(self, values: dict[str, Any]) -> None:
        task_family = values.get("task_family")
        if task_family:
            self.intent = str(task_family)
        task_subtype = values.get("task_subtype")
        if task_subtype:
            self.intent_pattern = str(task_subtype)

    def set_temporal_observations(self, value: dict[str, Any] | None) -> None:
        self.temporal_observations = deepcopy(value or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "intent_pattern": self.intent_pattern,
            "temporal_observations": deepcopy(self.temporal_observations),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "MemoryRuntimeContext":
        value = value or {}
        return cls(
            intent=value.get("intent"),
            intent_pattern=value.get("intent_pattern"),
            temporal_observations=deepcopy(value.get("temporal_observations", {})),
        )


class StateProfileBuilder:
    """Build phase-aware state-memory indexes and non-index profiles."""

    def build_index(
        self,
        *,
        state: Any,
        phase_id: str,
        skill_id: str,
    ) -> MemoryIndex:
        context = self._memory_context(state)
        intent = context.intent or self._spo_value(state, "P0", "task_family")
        if not intent:
            raise ValueError("P0A1 task intent is required for memory indexing")
        return MemoryIndex(
            phase_id=phase_id,
            skill_id=skill_id,
            intent=str(intent),
        )

    def build_profile(
        self,
        *,
        state: Any,
        required_spo: Iterable[str] = (),
        masked_action_space: Iterable[str] = (),
        active_profile_fields: Iterable[str] = (),
    ) -> StateMemoryProfile:
        context = self._memory_context(state)
        fields = tuple(active_profile_fields)
        if any(path.startswith("temporal.") for path in fields):
            fields = tuple(dict.fromkeys((*fields, *(f"temporal.{name}" for name in SAMPLING_FIELDS))))
        return StateMemoryProfile(
            canonical_spo=self._canonical_spo(state, required_spo),
            query=self._query_profile(state),
            schema=self._schema_profile(state),
            temporal=self._temporal_profile(state, context),
            masked_action_space=tuple(masked_action_space),
            active_profile_fields=fields,
        )

    def _canonical_spo(
        self,
        state: Any,
        references: Iterable[str],
    ) -> dict[str, Any]:
        result = {}
        for reference in references:
            reference = str(reference).strip()
            if not reference or "." not in reference:
                raise ValueError(
                    f"Canonical SPO reference must use PHASE.field: {reference!r}"
                )
            phase_id, field_name = reference.split(".", 1)
            if phase_id == "P0" and field_name in {"task_family", "task_subtype"}:
                continue
            result[reference] = self._spo_value(
                state,
                phase_id,
                field_name,
                default=UNSET,
            )
        return result

    def _query_profile(self, state: Any) -> QueryStructureProfile:
        frame = self._flat_query_frame(state)
        operator_store = self._operator_store(state)
        groups = _select_groups(frame.get("select"))
        expressions_by_group = [_split_expressions(group) for group in groups]
        expressions = [item for group in expressions_by_group for item in group]
        descriptor_index = _expression_descriptor_index(operator_store)
        descriptors = [
            descriptor_index.get(_normalize_expression(expression))
            for expression in expressions
        ]

        return QueryStructureProfile(
            select_shape=_select_shape(
                frame=frame,
                expressions=expressions,
                descriptors=descriptors,
                state=state,
            ),
            select_layout=SelectLayoutProfile(
                group_count=cardinality_bucket(len(groups)),
                expressions_per_group=_expression_width_bucket(expressions_by_group),
            ),
            select_type_family=_descriptor_type_family(descriptors),
            expression_depth=_descriptor_depth(descriptors),
            clause_mask=_clause_mask(frame),
        )

    def _schema_profile(self, state: Any) -> SchemaProfile:
        working_schema = self._working_schema(state)
        if not working_schema:
            return SchemaProfile()

        devices = working_schema.get("devices", [])
        device_records = [device for device in devices if isinstance(device, dict)]
        device_count = _safe_int(working_schema.get("device_count"))
        if device_count is None:
            device_count = len(devices) if isinstance(devices, list) else None

        names: set[str] = set()
        data_types: list[str] = []
        per_device_names: list[set[str]] = []
        derived_timeseries_count = 0
        for device in device_records:
            current_names: set[str] = set()
            for measurement in device.get("measurements", []) or []:
                if not isinstance(measurement, dict):
                    continue
                name = measurement.get("name") or measurement.get("measurement")
                if name:
                    normalized_name = str(name)
                    names.add(normalized_name)
                    current_names.add(normalized_name)
                    derived_timeseries_count += 1
                if measurement.get("data_type"):
                    data_types.append(str(measurement["data_type"]))
            per_device_names.append(current_names)

        timeseries_count = _safe_int(working_schema.get("timeseries_count"))
        if timeseries_count is None:
            timeseries_count = derived_timeseries_count if device_records else None

        measurement_count = len(names) if device_records else None
        return SchemaProfile(
            device_count=cardinality_bucket(device_count),
            measurement_name_count=cardinality_bucket(measurement_count),
            timeseries_count=cardinality_bucket(timeseries_count),
            measurement_sharing=_measurement_sharing(per_device_names),
            data_type_family=_data_type_family(data_types),
        )

    def _temporal_profile(
        self,
        state: Any,
        context: MemoryRuntimeContext,
    ) -> TemporalProfile:
        intent_pattern = context.intent_pattern or self._spo_value(
            state,
            "P0",
            "task_subtype",
        )
        ranges = _temporal_ranges(context.temporal_observations)
        observations_available = bool(context.temporal_observations)
        data_span = _data_span(
            ranges,
            observations_available=observations_available,
        )
        coverage_overlap = _coverage_overlap(
            ranges,
            observations_available=observations_available,
        )
        frame = self._flat_query_frame(state)
        temporal_mode = self._spo_value(state, "P2", "temporal_mode")
        query_scope, query_bounds = _query_scope_profile(
            frame.get("time"),
            temporal_mode=temporal_mode,
            data_ranges=ranges,
        )
        window_shape = _window_shape_profile(
            frame.get("time"),
            query_bounds=query_bounds,
            data_ranges=ranges,
        )
        return TemporalProfile(
            intent_pattern=str(intent_pattern) if intent_pattern else UNSET,
            data_span=data_span,
            coverage_overlap=coverage_overlap,
            query_scope=query_scope,
            window_shape=window_shape,
            **sampling_features(context.temporal_observations, query_bounds),
        )

    @staticmethod
    def _memory_context(state: Any) -> MemoryRuntimeContext:
        context = getattr(state, "memory_context", None)
        if isinstance(context, MemoryRuntimeContext):
            return context
        if isinstance(context, dict):
            return MemoryRuntimeContext.from_dict(context)
        return MemoryRuntimeContext()

    @staticmethod
    def _spo_value(
        state: Any,
        phase_id: str,
        field_name: str,
        default: Any = None,
    ) -> Any:
        if hasattr(state, "get_spo_value"):
            return state.get_spo_value(phase_id, field_name, default)
        if isinstance(state, dict):
            return (
                state.get("spo", {})
                .get(phase_id, {})
                .get(field_name, default)
            )
        return default

    @staticmethod
    def _flat_query_frame(state: Any) -> dict[str, Any]:
        query_frame = getattr(state, "query_frame", None)
        if query_frame is not None and hasattr(query_frame, "to_flat_dict"):
            return query_frame.to_flat_dict()
        if isinstance(state, dict):
            value = state.get("query_frame_flat", state.get("query_frame", {}))
            return deepcopy(value) if isinstance(value, dict) else {}
        return {}

    @staticmethod
    def _operator_store(state: Any) -> dict[str, Any]:
        if hasattr(state, "get_operator_store"):
            return state.get_operator_store()
        if isinstance(state, dict):
            value = state.get("operator_store", {})
            return deepcopy(value) if isinstance(value, dict) else {}
        return {}

    @staticmethod
    def _working_schema(state: Any) -> dict[str, Any] | None:
        schema = getattr(state, "schema", None)
        if schema is not None and hasattr(schema, "get_active_working_schema"):
            return schema.get_active_working_schema()
        if isinstance(state, dict):
            schema_context = state.get("schema", {})
            if isinstance(schema_context, dict):
                value = schema_context.get("working_schema")
                return deepcopy(value) if isinstance(value, dict) else None
        return None


def _select_groups(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _split_expressions(value: str) -> list[str]:
    expressions: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    for index, char in enumerate(value):
        if quote:
            if char == quote and (index == 0 or value[index - 1] != "\\"):
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        elif char == "," and depth == 0:
            item = value[start:index].strip()
            if item:
                expressions.append(item)
            start = index + 1
    item = value[start:].strip()
    if item:
        expressions.append(item)
    return expressions


def _normalize_expression(value: str) -> str:
    value = re.sub(r"\s+AS\s+[^,]+$", "", str(value).strip(), flags=re.I)
    return re.sub(r"\s+", "", value).lower()


def _expression_descriptor_index(operator_store: dict[str, Any]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for source, kind in (("SeriesExpr", "series"), ("AggExpr", "aggregate")):
        for descriptor in operator_store.get(source, []) or []:
            if not isinstance(descriptor, dict):
                continue
            name = descriptor.get("name") or descriptor.get("expression")
            if not name:
                continue
            current = deepcopy(descriptor)
            current["_kind"] = kind
            result[_normalize_expression(str(name))] = current
    return result


def _select_shape(
    *,
    frame: dict[str, Any],
    expressions: list[str],
    descriptors: list[dict | None],
    state: Any,
) -> str:
    if not expressions:
        return UNSET
    if frame.get("sql_query_mode") == "select_last":
        return "latest"
    kinds = set()
    for expression, descriptor in zip(expressions, descriptors):
        if descriptor is not None:
            if descriptor.get("_kind") == "aggregate":
                kinds.add("aggregate_expr")
            elif _safe_int(descriptor.get("op_level")) == 1:
                kinds.add("raw")
            else:
                kinds.add("series_expr")
            continue
        if re.fullmatch(r"[A-Za-z_][\w.]*", expression.strip()):
            kinds.add("raw")
        else:
            regime = StateProfileBuilder._spo_value(
                state,
                "P4",
                "operator_regime",
            )
            if regime in {"aggregate_query", "mixed_nested_query"}:
                kinds.add("aggregate_expr")
            elif regime == "timeseries_function_query":
                kinds.add("series_expr")
            else:
                kinds.add(UNKNOWN)
    return next(iter(kinds)) if len(kinds) == 1 else "mixed"


def _expression_width_bucket(groups: list[list[str]]) -> str:
    if not groups:
        return "zero"
    buckets = {cardinality_bucket(len(group)) for group in groups}
    return next(iter(buckets)) if len(buckets) == 1 else "mixed"


def _descriptor_type_family(descriptors: list[dict | None]) -> str:
    if not descriptors:
        return UNSET
    if any(descriptor is None for descriptor in descriptors):
        return UNKNOWN
    return _data_type_family(
        [str(descriptor.get("data_type")) for descriptor in descriptors]
    )


def _descriptor_depth(descriptors: list[dict | None]) -> str:
    if not descriptors:
        return UNSET
    levels = [
        _safe_int(descriptor.get("op_level"))
        for descriptor in descriptors
        if descriptor is not None
    ]
    if len(levels) != len(descriptors) or any(level is None for level in levels):
        return UNKNOWN
    maximum = max(levels)
    if maximum <= 1:
        return "atomic"
    if maximum == 2:
        return "shallow"
    if maximum <= 4:
        return "nested"
    return "deep"


def _clause_mask(frame: dict[str, Any]) -> int:
    mask = ClauseFlag(0)
    if _has_concrete_time(frame.get("time")):
        mask |= ClauseFlag.TIME
    if _nonempty(frame.get("group_by")):
        mask |= ClauseFlag.GROUP_BY
    if _nonempty(frame.get("where")):
        mask |= ClauseFlag.WHERE
    if _nonempty(frame.get("having")):
        mask |= ClauseFlag.HAVING
    if _active_mapping(frame.get("fill")):
        mask |= ClauseFlag.FILL
    if _active_mapping(frame.get("into")):
        mask |= ClauseFlag.INTO
    if _nonempty(frame.get("order_by")):
        mask |= ClauseFlag.ORDER_BY
    if any(frame.get(key) is not None for key in ("limit", "offset", "slimit", "soffset")):
        mask |= ClauseFlag.PAGINATION
    return int(mask)


def _has_concrete_time(value: Any) -> bool:
    if not _nonempty(value):
        return False
    if isinstance(value, dict):
        concrete_keys = set(value) - {"temporal_mode"}
        return bool(concrete_keys)
    text = str(value).strip().lower()
    return text not in {"no_time", "point_time", "latest_time"}


def _active_mapping(value: Any) -> bool:
    if not _nonempty(value):
        return False
    if isinstance(value, dict):
        mode = str(value.get("mode", value.get("fill_mode", ""))).lower()
        return mode not in {"none", "no_fill", "no_writeback"}
    return True


def _nonempty(value: Any) -> bool:
    return value not in (None, "", [], {})


def _measurement_sharing(per_device_names: list[set[str]]) -> str:
    if not per_device_names:
        return UNKNOWN
    if len(per_device_names) == 1:
        return NOT_APPLICABLE
    all_names = set().union(*per_device_names)
    if not all_names:
        return UNKNOWN
    shared_by_all = set.intersection(*per_device_names)
    if shared_by_all == all_names:
        return "full"
    if any(
        sum(name in names for names in per_device_names) > 1
        for name in all_names
    ):
        return "partial"
    return "none"


def _data_type_family(values: Iterable[str]) -> str:
    normalized = {str(value).upper() for value in values if value and value != "None"}
    if not normalized:
        return UNKNOWN
    families = set()
    for value in normalized:
        if value in NUMERIC_TYPES:
            families.add("numeric")
        elif value in TEXT_TYPES:
            families.add("text")
        elif value in BOOLEAN_TYPES:
            families.add("boolean")
        else:
            families.add("unknown")
    if families == {"numeric"}:
        return "numeric_only"
    if families == {"text"}:
        return "text_only"
    if families == {"boolean"}:
        return "boolean_only"
    if "unknown" in families:
        return UNKNOWN
    return "mixed"


def _temporal_ranges(observations: dict[str, Any]) -> list[tuple[int, int]]:
    ranges = []
    for measurement in observations.get("measurements", []) or []:
        if not isinstance(measurement, dict):
            continue
        start = _timestamp_ms(
            measurement.get("start_time_ms", measurement.get("start_time"))
        )
        end = _timestamp_ms(
            measurement.get("end_time_ms", measurement.get("end_time"))
        )
        if start is not None and end is not None and start <= end:
            ranges.append((start, end))
    return ranges


def _data_span(
    ranges: list[tuple[int, int]],
    *,
    observations_available: bool,
) -> str:
    if not ranges:
        return "empty" if observations_available else UNSET
    duration = max(end for _, end in ranges) - min(start for start, _ in ranges)
    if duration == 0:
        return "point"
    day_ms = 86_400_000
    if duration < day_ms:
        return "short"
    if duration <= 30 * day_ms:
        return "medium"
    return "long"


def _coverage_overlap(
    ranges: list[tuple[int, int]],
    *,
    observations_available: bool,
) -> str:
    if not ranges:
        return NOT_APPLICABLE if observations_available else UNSET
    if len(ranges) == 1:
        return "single"
    if max(start for start, _ in ranges) <= min(end for _, end in ranges):
        return "common"
    ordered = sorted(ranges)
    if any(current[0] <= previous[1] for previous, current in zip(ordered, ordered[1:])):
        return "partial"
    return "disjoint"


def _query_scope_profile(
    value: Any,
    *,
    temporal_mode: Any,
    data_ranges: list[tuple[int, int]],
) -> tuple[QueryScopeProfile, tuple[int | None, int | None] | None]:
    if temporal_mode == "no_time":
        return QueryScopeProfile(NONE, NOT_APPLICABLE, NOT_APPLICABLE), None
    text = _temporal_text(value)
    if not text:
        if temporal_mode in {"no_time", "latest_time"}:
            return QueryScopeProfile(NONE, NOT_APPLICABLE, NOT_APPLICABLE), None
        if temporal_mode in {"point_time", "range_time", "window_time"}:
            return QueryScopeProfile(UNKNOWN, UNKNOWN, UNKNOWN), None
        return QueryScopeProfile(), None

    bounds = _query_bounds(text)
    if bounds is None:
        return QueryScopeProfile(UNKNOWN, UNKNOWN, UNKNOWN), None
    start, end = bounds
    if not data_ranges:
        return QueryScopeProfile(UNKNOWN, UNKNOWN, UNKNOWN), bounds

    hits = sum(
        _intervals_overlap(start, end, data_start, data_end)
        for data_start, data_end in data_ranges
    )
    overlap = "none" if hits == 0 else "all" if hits == len(data_ranges) else "some"
    global_start = min(item[0] for item in data_ranges)
    global_end = max(item[1] for item in data_ranges)
    span_ratio = _span_ratio(start, end, global_start, global_end)
    position = _query_position(start, end, global_start, global_end)
    return QueryScopeProfile(overlap, span_ratio, position), bounds


def _window_shape_profile(
    value: Any,
    *,
    query_bounds: tuple[int | None, int | None] | None,
    data_ranges: list[tuple[int, int]],
) -> WindowShapeProfile:
    text = _temporal_text(value)
    if not text or "group by" not in text.lower():
        return WindowShapeProfile(NOT_APPLICABLE, NOT_APPLICABLE)
    durations = [
        _duration_ms(amount, unit)
        for amount, unit in DURATION_RE.findall(text)
    ]
    durations = [value for value in durations if value is not None]
    if not durations:
        return WindowShapeProfile(UNKNOWN, UNKNOWN)
    interval = durations[0]
    step = durations[1] if len(durations) > 1 else interval
    if step == interval:
        step_relation = "tumbling"
    elif step < interval:
        step_relation = "overlapping"
    else:
        step_relation = "gapped"

    start = end = None
    if query_bounds is not None:
        start, end = query_bounds
    if (start is None or end is None) and data_ranges:
        start = min(item[0] for item in data_ranges)
        end = max(item[1] for item in data_ranges)
    if start is None or end is None or step <= 0:
        window_count = UNKNOWN
    else:
        window_count = cardinality_bucket(max(1, int((end - start) / step)))
    return WindowShapeProfile(window_count, step_relation)


def _temporal_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        pieces = [str(item) for key, item in value.items() if key != "temporal_mode"]
        return " ".join(pieces).strip()
    if isinstance(value, list):
        return " ".join(str(item) for item in value).strip()
    return ""


def _query_bounds(text: str) -> tuple[int | None, int | None] | None:
    timestamps = [_timestamp_ms(value) for value in ISO_TIMESTAMP_RE.findall(text)]
    timestamps = [value for value in timestamps if value is not None]
    if len(timestamps) >= 2:
        return min(timestamps[0], timestamps[1]), max(timestamps[0], timestamps[1])

    start = end = None
    for operator, raw_value in TIME_COMPARISON_RE.findall(text):
        timestamp = _timestamp_ms(raw_value)
        if timestamp is None:
            continue
        if operator in {">", ">="}:
            start = timestamp
        elif operator in {"<", "<="}:
            end = timestamp
        else:
            start = end = timestamp
    if start is not None or end is not None:
        return start, end
    return None


def _timestamp_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        raw = int(value)
        return raw * 1000 if abs(raw) < 10_000_000_000 else raw
    text = str(value).strip().strip("'\"")
    if text.isdigit():
        raw = int(text)
        return raw * 1000 if len(text) <= 10 else raw
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _intervals_overlap(
    query_start: int | None,
    query_end: int | None,
    data_start: int,
    data_end: int,
) -> bool:
    effective_start = data_start if query_start is None else query_start
    effective_end = data_end if query_end is None else query_end
    return effective_start <= data_end and effective_end >= data_start


def _span_ratio(
    query_start: int | None,
    query_end: int | None,
    data_start: int,
    data_end: int,
) -> str:
    if query_start is None or query_end is None or data_end <= data_start:
        return NOT_APPLICABLE
    ratio = (query_end - query_start) / (data_end - data_start)
    if ratio <= 0.1:
        return "narrow"
    if ratio <= 0.5:
        return "medium"
    if ratio <= 1:
        return "broad"
    return "oversized"


def _query_position(
    query_start: int | None,
    query_end: int | None,
    data_start: int,
    data_end: int,
) -> str:
    if query_start is None and query_end is None:
        return NOT_APPLICABLE
    if query_start is None:
        return "head"
    if query_end is None:
        return "tail"
    if query_end < data_start or query_start > data_end:
        return "outside"
    if query_start <= data_start and query_end >= data_end:
        return "full"
    if data_end <= data_start:
        return "middle"
    midpoint_ratio = (((query_start + query_end) / 2) - data_start) / (data_end - data_start)
    if midpoint_ratio < 1 / 3:
        return "head"
    if midpoint_ratio > 2 / 3:
        return "tail"
    return "middle"


def _duration_ms(amount: str, unit: str) -> int | None:
    multipliers = {
        "ms": 1,
        "s": 1000,
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 7 * 86_400_000,
        "mo": 30 * 86_400_000,
        "y": 365 * 86_400_000,
    }
    multiplier = multipliers.get(unit.lower())
    if multiplier is None:
        return None
    return int(float(amount) * multiplier)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
