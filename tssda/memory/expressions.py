"""Small expression reuse layer sharing state memory's validation lifecycle.

Expressions are stored as typed templates over raw working-schema measurements.
Only expressions present in a validated final SQL are admitted, including nested
subexpressions. No model calls or database probes are needed for retrieval.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import re
import random
from typing import Any, Iterable

from .matcher import MemoryMatchConfig, score_profiles, temporal_rejection_reasons
from .models import MemoryStatistics, StateMemoryProfile, stable_fingerprint
from .temporal import inputs_have_common_timestamps
from .profiles import StateProfileBuilder


# Quoted literals stay opaque: rebinding must never alter function attributes.
TOKEN_RE = re.compile(
    r"'(?:''|\\.|[^'])*'|\"(?:\"\"|\\.|[^\"])*\"|`(?:``|[^`])*`"
    r"|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    r"|[^\W\d][\w$]*(?:\.[^\W\d][\w$]*)*|>=|<=|!=|<>|==|&&|\|\||\S",
    re.UNICODE,
)
IDENTIFIER_RE = re.compile(r"(?:[^\W\d][\w$]*)(?:\.[^\W\d][\w$]*)*\Z")
MEASUREMENT_RE = re.compile(r"[^\W\d][\w$]*\Z")
DATA_TYPES = frozenset({"INT32", "INT64", "FLOAT", "DOUBLE", "BOOLEAN", "TEXT", "STRING"})
KEYWORDS = frozenset({"AND", "OR", "NOT", "TRUE", "FALSE", "NULL", "IS", "LIKE", "REGEXP"})
KINDS_BY_REGIME = {
    "timeseries_function_query": "SeriesExpr",
    "aggregate_query": "AggExpr",
    "mixed_nested_query": "AggExpr",
}


@dataclass
class ExpressionMemoryRecord:
    context: dict[str, Any]
    kind: str
    template: str
    inputs: list[dict[str, str]]
    data_type: str
    op_level: int
    statistics: MemoryStatistics = field(default_factory=MemoryStatistics)

    @property
    def index_key(self) -> tuple[str, str, str]:
        return (self.context["intent"], self.context["operator_regime"], self.kind)

    @property
    def record_key(self) -> str:
        value = self.to_dict()
        value.pop("statistics")
        # Measurement names are the available semantic identity. Do not merge
        # unrelated operands solely because their types happen to match.
        return stable_fingerprint(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": deepcopy(self.context), "kind": self.kind,
            "template": self.template, "inputs": deepcopy(self.inputs),
            "data_type": self.data_type, "op_level": self.op_level,
            "statistics": self.statistics.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExpressionMemoryRecord":
        return cls(
            context=deepcopy(value["context"]), kind=value["kind"],
            template=value["template"], inputs=deepcopy(value["inputs"]),
            data_type=value["data_type"], op_level=int(value["op_level"]),
            statistics=MemoryStatistics.from_dict(value.get("statistics", {})),
        )


class ExpressionMemoryStore:
    """Indexed records persisted inside the existing state-memory JSON file."""

    def __init__(self, records: Iterable[ExpressionMemoryRecord] = ()):
        self._index: dict[tuple[str, str, str], dict[str, ExpressionMemoryRecord]] = {}
        for record in records:
            self.add(record)

    def add(self, record: ExpressionMemoryRecord) -> None:
        bucket = self._index.setdefault(record.index_key, {})
        existing = bucket.get(record.record_key)
        if existing is None:
            bucket[record.record_key] = ExpressionMemoryRecord.from_dict(record.to_dict())
        else:
            existing.statistics.merge(record.statistics)

    def candidates(self, context: dict[str, Any], kind: str) -> list[ExpressionMemoryRecord]:
        key = (context["intent"], context["operator_regime"], kind)
        return [ExpressionMemoryRecord.from_dict(record.to_dict())
                for record in self._index.get(key, {}).values()]

    def settle(self, record: ExpressionMemoryRecord, *, validated: bool, passed: bool,
               used: bool, config: MemoryMatchConfig) -> bool:
        bucket = self._index.get(record.index_key, {})
        stored = bucket.get(record.record_key)
        if stored is None:
            return False
        stored.statistics.record_reuse()
        if validated and used:
            stored.statistics.record_validation(success=passed)
            if not passed and stored.statistics.confidence(
                alpha=config.reliability_alpha, beta=config.reliability_beta,
            ) <= config.deletion_confidence_threshold:
                del bucket[record.record_key]
        return True

    @property
    def record_count(self) -> int:
        return sum(len(bucket) for bucket in self._index.values())

    def to_list(self) -> list[dict[str, Any]]:
        records = [record for bucket in self._index.values() for record in bucket.values()]
        return [record.to_dict() for record in sorted(records, key=lambda item: item.record_key)]


def _context(state: Any) -> dict[str, Any]:
    # Reuse the same observed temporal features and Sim as state-level memory.
    builder = StateProfileBuilder()
    schema = builder._schema_profile(state)
    return {
        "intent": state.memory_context.intent or state.get_spo_value("P0", "task_family"),
        "intent_pattern": state.memory_context.intent_pattern or state.get_spo_value("P0", "task_subtype"),
        "operator_regime": state.get_spo_value("P4", "operator_regime"),
        "temporal_mode": state.get_spo_value("P2", "temporal_mode"),
        "group_paradigm": state.get_spo_value("P3", "group_paradigm"),
        "segment_axis": state.get_spo_value("P3", "segment_axis"),
        "group_axis": state.get_spo_value("P3", "group_axis"),
        "measurement_sharing": schema.measurement_sharing,
        "profile": builder.build_profile(state=state, active_profile_fields=(
            "schema.measurement_sharing", "schema.data_type_family",
            "temporal.intent_pattern", "temporal.data_span", "temporal.coverage_overlap",
            "temporal.query_scope.data_overlap", "temporal.query_scope.span_ratio",
            "temporal.query_scope.position", "temporal.window_shape.window_count",
            "temporal.window_shape.step_relation",
        )).to_dict(),
    }


def _raw_inputs(state: Any) -> dict[str, str]:
    """Resolve only unambiguous measurement types in the final P1 scope."""
    schema = state.schema.get_active_working_schema() or {}
    types: dict[str, set[str]] = {}
    for device in schema.get("devices", []):
        for measurement in device.get("measurements", []) or []:
            name = measurement.get("name") or measurement.get("measurement")
            if name:
                types.setdefault(str(name), set()).add(str(measurement.get("data_type", "")).upper())
    # Flat working schemas are supported by OperatorStore as well.
    if not types:
        for measurement in schema.get("measurements", []) or []:
            if isinstance(measurement, dict) and measurement.get("name"):
                types.setdefault(measurement["name"], set()).add(str(measurement.get("data_type", "")).upper())
    return {name: next(iter(values)) for name, values in types.items()
            if len(values) == 1 and next(iter(values)) in DATA_TYPES}


def _tokens(expression: str) -> list[str]:
    return TOKEN_RE.findall(expression)


def build_expression_record(state: Any, kind: str, descriptor: dict[str, Any]) -> ExpressionMemoryRecord | None:
    """Conservatively template a concrete, fully instantiated expression."""
    expression = str(descriptor.get("name") or descriptor.get("expression") or "")
    raw = _raw_inputs(state)
    tokens = _tokens(expression)
    if not tokens or kind not in {"SeriesExpr", "AggExpr"}:
        return None
    data_type = descriptor.get("data_type")
    if not isinstance(data_type, str) or data_type not in DATA_TYPES:
        return None
    inputs: list[dict[str, str]] = []
    slots: dict[str, str] = {}
    for index, token in enumerate(tokens):
        name = token[1:-1].replace("``", "`") if token.startswith("`") else token
        if token.startswith(("'", '"')):
            continue
        if index + 1 < len(tokens) and tokens[index + 1] == "(":
            # Abstract P4 placeholders must never reach persistent memory.
            if name.upper() in {"AGG", "TSGF", "UNARYOP", "BINARYOP"} or "." in name:
                return None
            continue
        if name in raw:
            if name not in slots:
                slots[name] = f"__mem_input_{len(inputs)}__"
                inputs.append({"name": name, "data_type": raw[name]})
            tokens[index] = slots[name]
        elif (IDENTIFIER_RE.fullmatch(name) or token.startswith("`")) and name.upper() not in KEYWORDS:
            return None
    try:
        level = int(descriptor["op_level"])
    except (KeyError, ValueError, TypeError):
        return None
    if not inputs or level < 1 or (kind == "SeriesExpr" and len(tokens) == 1):
        return None
    context = _context(state)
    if not context["intent"] or not context["operator_regime"]:
        return None
    return ExpressionMemoryRecord(context, kind, " ".join(tokens), inputs, data_type, level)


def _bind(record: ExpressionMemoryRecord, raw: dict[str, str]) -> dict[str, Any] | None:
    bindings: dict[str, str] = {}
    # Exact measurement name and type are a conservative semantic binding.
    # With no ontology/aliases in State, unrelated same-type names are rejected.
    for index, item in enumerate(record.inputs):
        if raw.get(item["name"]) == item["data_type"]:
            bindings[f"__mem_input_{index}__"] = item["name"]
    for index, item in enumerate(record.inputs):
        slot = f"__mem_input_{index}__"
        if slot in bindings:
            continue
        return None
    def render(token: str) -> str:
        if token not in bindings:
            return token
        name = bindings[token]
        return name if MEASUREMENT_RE.fullmatch(name) and name.upper() not in KEYWORDS else "`" + name.replace("`", "``") + "`"
    return {"name": " ".join(render(token) for token in _tokens(record.template)),
            "data_type": record.data_type, "op_level": record.op_level}


def _comparison_tokens(expression: str) -> list[str]:
    tokens = _tokens(expression)
    return [token.upper() if index + 1 < len(tokens) and tokens[index + 1] == "(" and IDENTIFIER_RE.fullmatch(token)
            else token[1:-1].replace("``", "`") if token.startswith("`") else token
            for index, token in enumerate(tokens)]


def _used_in_final_sql(state: Any, expression: str, *, allow_provisional: bool = False) -> bool:
    needle = _comparison_tokens(expression)
    # P10 may repair or replace SELECT expressions: validate against the actual
    # accepted SQL, not the provisional frame. Token comparison preserves strings.
    sqls = [sample.get("sql", "") for sample in state.final_samples]
    if not sqls and allow_provisional:
        select = state.query_frame.select
        sqls = select if isinstance(select, list) else [select or ""]
    for sql in sqls:
        tokens = _comparison_tokens(sql)
        if any(tokens[start:start + len(needle)] == needle for start in range(len(tokens) - len(needle) + 1)):
            return True
    return False


class ExpressionMemoryController:
    def __init__(self, store: ExpressionMemoryStore, config: MemoryMatchConfig, *, read_enabled: bool = True):
        self.store, self.config, self.read_enabled = store, config, read_enabled
        self._random = random.Random(config.random_seed)
        self._pending: dict[str, tuple[ExpressionMemoryRecord, Any, str]] = {}
        self._reused: dict[str, tuple[ExpressionMemoryRecord, Any, str]] = {}

    def begin_trajectory(self) -> None:
        self._pending.clear()
        self._reused.clear()

    def populate(self, state: Any, *, kind: str, max_level: int) -> dict[str, Any]:
        context = _context(state)
        trace: dict[str, Any] = {"enabled": True, "read_enabled": self.read_enabled,
                                 "kind": kind, "hit": False, "record_ids": [], "expressions": [],
                                 "candidate_count": 0, "profile_key": stable_fingerprint(context),
                                 "index": {"intent": context["intent"],
                                           "operator_regime": context["operator_regime"], "kind": kind},
                                 "max_operator_level": max_level}
        if not self.read_enabled:
            return trace
        candidates = self.store.candidates(context, kind)
        trace["candidate_count"] = len(candidates)
        raw = _raw_inputs(state)
        config = self.config
        eligible = []
        profile = StateMemoryProfile.from_dict(context["profile"])
        trace["temporal_profile"] = profile.temporal.to_matching_dict()
        for record in candidates:
            confidence = record.statistics.confidence(alpha=config.reliability_alpha, beta=config.reliability_beta)
            if ({k: v for k, v in record.context.items() if k != "profile"}
                    != {k: v for k, v in context.items() if k != "profile"}
                    or record.op_level > max_level
                    or record.statistics.ns < config.min_validation_successes
                    or confidence < config.min_reliability):
                continue
            if "profile" not in record.context:
                continue
            remembered = StateMemoryProfile.from_dict(record.context["profile"])
            similarity = score_profiles(profile, remembered)
            if (remembered.profile_version != profile.profile_version
                    or temporal_rejection_reasons(profile.temporal, remembered.temporal)
                    or similarity.similarity < config.min_profile_similarity
                    or similarity.coverage < config.min_evidence_coverage):
                continue
            bound = _bind(record, raw)
            if bound is not None and (len(record.inputs) < 2 or inputs_have_common_timestamps(
                state.memory_context.temporal_observations, [item["name"] for item in record.inputs],
            )):
                score = similarity.similarity * confidence / (1 + config.reuse_penalty * record.statistics.nr)
                eligible.append((score, record, bound, similarity.similarity))
        existing = {tuple(_comparison_tokens(item.get("name") or item.get("expression") or ""))
                    for item in state.get_operator_store().get(kind, [])}
        eligible = [item for item in eligible if tuple(_comparison_tokens(item[2]["name"])) not in existing]
        eligible.sort(key=lambda item: (-item[0], item[1].record_key))
        eligible = eligible[:config.top_k]
        total = sum(item[0] for item in eligible)
        trace["matches"] = [{"record_id": r.record_key, "profile_similarity": sim,
                             "rank_score": score, "selection_probability": score / total}
                            for score, r, _, sim in eligible] if total else []
        if total:
            _, record, bound, _ = self._random.choices(eligible, weights=[item[0] for item in eligible], k=1)[0]
            state.update_operator_store({f"append_{kind}": [bound]})
            self._reused[record.record_key] = (record, state, bound["name"])
            trace["record_ids"].append(record.record_key)
            trace["expressions"].append(deepcopy(bound))
        trace["hit"] = bool(trace["expressions"])
        return trace

    def stage(self, state: Any, kind: str, descriptors: list[dict[str, Any]]) -> None:
        for descriptor in descriptors:
            record = build_expression_record(state, kind, descriptor)
            if record is not None and record.record_key not in self._reused:
                self._pending[record.record_key] = (record, state, str(descriptor.get("name") or descriptor["expression"]))

    def finish_trajectory(self, validation_result: Any) -> bool:
        changed = False
        for record, state, expression in self._reused.values():
            changed = self.store.settle(
                record, validated=validation_result.validated, passed=validation_result.passed,
                used=_used_in_final_sql(state, expression, allow_provisional=not validation_result.passed),
                config=self.config,
            ) or changed
        if validation_result.passed:
            for record, state, expression in self._pending.values():
                if _used_in_final_sql(state, expression):
                    record.statistics.record_validation(success=True)
                    self.store.add(record)
                    changed = True
        self.begin_trajectory()
        return changed
