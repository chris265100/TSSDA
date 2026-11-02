"""Core data structures for TSSDA state-level reuse memory."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntFlag
import hashlib
import json
from typing import Any


PROFILE_VERSION = 2

UNSET = "UNSET"
NONE = "NONE"
UNKNOWN = "UNKNOWN"
NOT_APPLICABLE = "NOT_APPLICABLE"


class ClauseFlag(IntFlag):
    """Compact QueryFrame clause-presence bits."""

    TIME = 1 << 0
    GROUP_BY = 1 << 1
    WHERE = 1 << 2
    HAVING = 1 << 3
    FILL = 1 << 4
    INTO = 1 << 5
    ORDER_BY = 1 << 6
    PAGINATION = 1 << 7


def canonical_json(value: Any) -> str:
    """Return deterministic JSON for fingerprints and persistence."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_fingerprint(value: Any) -> str:
    """Return a stable SHA-256 fingerprint for a JSON-compatible value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def action_space_signature(action_space: list[str] | tuple[str, ...]) -> str:
    """Fingerprint a masked action set independently of candidate order."""

    normalized = sorted({str(value).strip() for value in action_space if value})
    return stable_fingerprint(normalized)


def cardinality_bucket(value: int | None) -> str:
    """Map an exact count to the shared zero/one/few/many vocabulary."""

    if value is None:
        return UNKNOWN
    if value <= 0:
        return "zero"
    if value == 1:
        return "one"
    if value <= 4:
        return "few"
    return "many"


TEMPORAL_SIGNATURE_WEIGHTS = {
    "temporal.intent_pattern": 0.08,
    "temporal.data_span": 0.04,
    "temporal.coverage_overlap": 0.06,
    "temporal.sampling_interval": 0.05,
    "temporal.sampling_regularity": 0.04,
    "temporal.timestamp_alignment": 0.08,
    "temporal.point_distribution": 0.05,
    "temporal.query_scope.data_overlap": 0.06,
    "temporal.query_scope.span_ratio": 0.04,
    "temporal.query_scope.position": 0.035,
    "temporal.window_shape.window_count": 0.025,
    "temporal.window_shape.step_relation": 0.04,
}


def temporal_signature_similarity(
    left: dict[str, Any],
    right: dict[str, Any],
) -> float:
    """Compare two normalized failure contexts with temporal-aware weights."""

    paths = set(left) | set(right)
    if not paths:
        return 0.0
    total_weight = sum(TEMPORAL_SIGNATURE_WEIGHTS.get(path, 0.01) for path in paths)
    matched_weight = 0.0
    for path in paths:
        if path not in left or path not in right:
            continue
        weight = TEMPORAL_SIGNATURE_WEIGHTS.get(path, 0.01)
        matched_weight += weight * _temporal_feature_similarity(
            path,
            left[path],
            right[path],
        )
    return matched_weight / total_weight if total_weight else 0.0


def _temporal_feature_similarity(path: str, left: Any, right: Any) -> float:
    if canonical_json(left) == canonical_json(right):
        return 1.0
    if path in {
        "temporal.data_span",
        "temporal.query_scope.span_ratio",
    }:
        order = (
            ("point", "short", "medium", "long")
            if path == "temporal.data_span"
            else ("narrow", "medium", "broad", "oversized")
        )
        return _ordered_temporal_similarity(left, right, order, floor=0.1)
    if path == "temporal.window_shape.window_count":
        if "zero" in {left, right}:
            return 0.0
        return _ordered_temporal_similarity(
            left,
            right,
            ("one", "few", "many"),
            floor=0.25,
        )
    pairs = {str(left), str(right)}
    if path == "temporal.coverage_overlap":
        if pairs == {"single", "common"}:
            return 0.8
        if pairs in ({"common", "partial"}, {"single", "partial"}):
            return 0.6
        if pairs == {"partial", "disjoint"}:
            return 0.25
    elif path == "temporal.query_scope.data_overlap":
        if pairs == {"all", "some"}:
            return 0.65
        if pairs == {"some", "none"}:
            return 0.2
    elif path == "temporal.query_scope.position":
        if pairs in ({"head", "middle"}, {"middle", "tail"}):
            return 0.55
        if "full" in pairs and "outside" not in pairs:
            return 0.4
    elif path == "temporal.window_shape.step_relation":
        if pairs == {"tumbling", "overlapping"}:
            return 0.55
        if "gapped" in pairs:
            return 0.1
    return 0.0


def _ordered_temporal_similarity(
    left: Any,
    right: Any,
    order: tuple[str, ...],
    *,
    floor: float,
) -> float:
    positions = {value: index for index, value in enumerate(order)}
    if left not in positions or right not in positions:
        return 0.0
    distance = abs(positions[left] - positions[right])
    return max(floor, 1.0 - 0.3 * distance)


@dataclass(frozen=True)
class MemoryIndex:
    """Primary lookup index: phase, skill, and P0A1 task intent."""

    phase_id: str
    skill_id: str
    intent: str

    def __post_init__(self) -> None:
        for field_name in ("phase_id", "skill_id", "intent"):
            if not str(getattr(self, field_name, "")).strip():
                raise ValueError(f"MemoryIndex.{field_name} must not be empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "phase_id": self.phase_id,
            "skill_id": self.skill_id,
            "intent": self.intent,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryIndex":
        return cls(
            phase_id=str(value["phase_id"]),
            skill_id=str(value["skill_id"]),
            intent=str(value["intent"]),
        )


@dataclass
class SelectLayoutProfile:
    group_count: str = UNSET
    expressions_per_group: str = UNSET

    def to_dict(self) -> dict[str, str]:
        return {
            "group_count": self.group_count,
            "expressions_per_group": self.expressions_per_group,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "SelectLayoutProfile":
        value = value or {}
        return cls(
            group_count=str(value.get("group_count", UNSET)),
            expressions_per_group=str(
                value.get("expressions_per_group", UNSET)
            ),
        )


@dataclass
class QueryStructureProfile:
    """Schema-independent structural summary of the current QueryFrame."""

    select_shape: str = UNSET
    select_layout: SelectLayoutProfile = field(default_factory=SelectLayoutProfile)
    select_type_family: str = UNSET
    expression_depth: str = UNSET
    clause_mask: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "select_shape": self.select_shape,
            "select_layout": self.select_layout.to_dict(),
            "select_type_family": self.select_type_family,
            "expression_depth": self.expression_depth,
            "clause_mask": int(self.clause_mask),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "QueryStructureProfile":
        value = value or {}
        return cls(
            select_shape=str(value.get("select_shape", UNSET)),
            select_layout=SelectLayoutProfile.from_dict(value.get("select_layout")),
            select_type_family=str(value.get("select_type_family", UNSET)),
            expression_depth=str(value.get("expression_depth", UNSET)),
            clause_mask=int(value.get("clause_mask", 0)),
        )


@dataclass
class SchemaProfile:
    """Compact cardinality and type profile of the active working schema."""

    device_count: str = UNSET
    measurement_name_count: str = UNSET
    timeseries_count: str = UNSET
    measurement_sharing: str = UNSET
    data_type_family: str = UNSET

    def to_dict(self) -> dict[str, str]:
        return {
            "device_count": self.device_count,
            "measurement_name_count": self.measurement_name_count,
            "timeseries_count": self.timeseries_count,
            "measurement_sharing": self.measurement_sharing,
            "data_type_family": self.data_type_family,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "SchemaProfile":
        value = value or {}
        return cls(
            device_count=str(value.get("device_count", UNSET)),
            measurement_name_count=str(
                value.get("measurement_name_count", UNSET)
            ),
            timeseries_count=str(value.get("timeseries_count", UNSET)),
            measurement_sharing=str(value.get("measurement_sharing", UNSET)),
            data_type_family=str(value.get("data_type_family", UNSET)),
        )


@dataclass
class QueryScopeProfile:
    data_overlap: str = UNSET
    span_ratio: str = UNSET
    position: str = UNSET

    def to_dict(self) -> dict[str, str]:
        return {
            "data_overlap": self.data_overlap,
            "span_ratio": self.span_ratio,
            "position": self.position,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "QueryScopeProfile":
        value = value or {}
        return cls(
            data_overlap=str(value.get("data_overlap", UNSET)),
            span_ratio=str(value.get("span_ratio", UNSET)),
            position=str(value.get("position", UNSET)),
        )


@dataclass
class WindowShapeProfile:
    window_count: str = UNSET
    step_relation: str = UNSET

    def to_dict(self) -> dict[str, str]:
        return {
            "window_count": self.window_count,
            "step_relation": self.step_relation,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "WindowShapeProfile":
        value = value or {}
        return cls(
            window_count=str(value.get("window_count", UNSET)),
            step_relation=str(value.get("step_relation", UNSET)),
        )


@dataclass
class TemporalFailureContext:
    """One normalized temporal context in which a reuse was invalid."""

    signature: dict[str, Any] = field(default_factory=dict)
    failures: int = 1

    def __post_init__(self) -> None:
        self.signature = deepcopy(_canonical_value(self.signature))
        if not self.signature:
            raise ValueError("Temporal failure context signature must not be empty")
        if any(not key.startswith("temporal.") for key in self.signature):
            raise ValueError(
                "Temporal failure context may contain only temporal profile fields"
            )
        if self.failures < 1:
            raise ValueError("Temporal failure count must be positive")

    @property
    def context_key(self) -> str:
        return stable_fingerprint(self.signature)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_key": self.context_key,
            "signature": deepcopy(self.signature),
            "failures": self.failures,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TemporalFailureContext":
        return cls(
            signature=deepcopy(value.get("signature", {})),
            failures=int(value.get("failures", 1)),
        )


@dataclass
class TemporalProfile:
    """Incremental temporal intent and data/query relationship profile."""

    intent_pattern: str = UNSET
    data_span: str = UNSET
    coverage_overlap: str = UNSET
    query_scope: QueryScopeProfile = field(default_factory=QueryScopeProfile)
    window_shape: WindowShapeProfile = field(default_factory=WindowShapeProfile)
    failure_contexts: list[TemporalFailureContext] = field(default_factory=list)
    sampling_interval: str = UNKNOWN
    sampling_regularity: str = UNKNOWN
    timestamp_alignment: str = UNKNOWN
    point_distribution: str = UNKNOWN

    def __post_init__(self) -> None:
        self.failure_contexts = [
            value
            if isinstance(value, TemporalFailureContext)
            else TemporalFailureContext.from_dict(value)
            for value in self.failure_contexts
        ]

    def to_matching_dict(self) -> dict[str, Any]:
        """Return the positive temporal prototype used for similarity scoring."""

        return {
            "intent_pattern": self.intent_pattern,
            "data_span": self.data_span,
            "coverage_overlap": self.coverage_overlap,
            "sampling_interval": self.sampling_interval,
            "sampling_regularity": self.sampling_regularity,
            "timestamp_alignment": self.timestamp_alignment,
            "point_distribution": self.point_distribution,
            "query_scope": self.query_scope.to_dict(),
            "window_shape": self.window_shape.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_matching_dict()
        value["failure_contexts"] = [
            context.to_dict() for context in self.failure_contexts
        ]
        return value

    def record_failure_context(
        self,
        signature: dict[str, Any],
        *,
        max_contexts: int = 8,
        similarity_threshold: float = 0.95,
    ) -> bool:
        """Add or reinforce one negative temporal applicability boundary."""

        if max_contexts < 1:
            raise ValueError("max_contexts must be positive")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        candidate = TemporalFailureContext(signature=signature)
        if self.failure_contexts:
            best_context = max(
                self.failure_contexts,
                key=lambda context: temporal_signature_similarity(
                    context.signature,
                    candidate.signature,
                ),
            )
            if temporal_signature_similarity(
                best_context.signature,
                candidate.signature,
            ) >= similarity_threshold:
                best_context.failures += 1
                return True
        self.failure_contexts.append(candidate)
        if len(self.failure_contexts) > max_contexts:
            # Keep repeatedly observed boundaries and prefer retaining recent
            # evidence when counts tie.
            remove_index = min(
                range(len(self.failure_contexts)),
                key=lambda index: (
                    self.failure_contexts[index].failures,
                    index,
                ),
            )
            del self.failure_contexts[remove_index]
        return True

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "TemporalProfile":
        value = value or {}
        return cls(
            intent_pattern=str(value.get("intent_pattern", UNSET)),
            data_span=str(value.get("data_span", UNSET)),
            coverage_overlap=str(value.get("coverage_overlap", UNSET)),
            sampling_interval=str(value.get("sampling_interval", UNKNOWN)),
            sampling_regularity=str(value.get("sampling_regularity", UNKNOWN)),
            timestamp_alignment=str(value.get("timestamp_alignment", UNKNOWN)),
            point_distribution=str(value.get("point_distribution", UNKNOWN)),
            query_scope=QueryScopeProfile.from_dict(value.get("query_scope")),
            window_shape=WindowShapeProfile.from_dict(value.get("window_shape")),
            failure_contexts=[
                TemporalFailureContext.from_dict(item)
                for item in value.get("failure_contexts", [])
                if isinstance(item, dict) and item.get("signature")
            ],
        )


@dataclass
class StateMemoryProfile:
    """All non-index state features used for compatibility and hit scoring."""

    canonical_spo: dict[str, Any] = field(default_factory=dict)
    query: QueryStructureProfile = field(default_factory=QueryStructureProfile)
    schema: SchemaProfile = field(default_factory=SchemaProfile)
    temporal: TemporalProfile = field(default_factory=TemporalProfile)
    masked_action_space: tuple[str, ...] = field(default_factory=tuple)
    active_profile_fields: tuple[str, ...] = field(default_factory=tuple)
    profile_version: int = PROFILE_VERSION

    def __post_init__(self) -> None:
        self.canonical_spo = deepcopy(_canonical_value(self.canonical_spo))
        self.masked_action_space = tuple(
            sorted({str(value).strip() for value in self.masked_action_space if value})
        )
        self.active_profile_fields = tuple(
            sorted(
                {
                    str(value).strip()
                    for value in self.active_profile_fields
                    if value
                }
            )
        )

    @property
    def action_space_key(self) -> str:
        return action_space_signature(self.masked_action_space)

    @property
    def profile_key(self) -> str:
        return stable_fingerprint(self.to_matching_dict())

    def to_matching_dict(self) -> dict[str, Any]:
        """Return only the profile dimensions active for this decision skill."""

        value = self.to_dict()
        # Learned negative applicability boundaries govern hard rejection but
        # are not part of the positive prototype fingerprint or similarity.
        value["temporal"] = self.temporal.to_matching_dict()
        if not self.active_profile_fields:
            return value
        features = {}
        for path in self.active_profile_fields:
            current: Any = value
            for part in path.split("."):
                if not isinstance(current, dict) or part not in current:
                    raise ValueError(f"Unknown active profile field: {path!r}")
                current = current[part]
            features[path] = deepcopy(current)
        return {
            "profile_version": self.profile_version,
            "canonical_spo": deepcopy(self.canonical_spo),
            "masked_action_space": list(self.masked_action_space),
            "active_profile_fields": list(self.active_profile_fields),
            "features": features,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_version": self.profile_version,
            "canonical_spo": deepcopy(self.canonical_spo),
            "query": self.query.to_dict(),
            "schema": self.schema.to_dict(),
            "temporal": self.temporal.to_dict(),
            "masked_action_space": list(self.masked_action_space),
            "active_profile_fields": list(self.active_profile_fields),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StateMemoryProfile":
        return cls(
            profile_version=int(value.get("profile_version", PROFILE_VERSION)),
            canonical_spo=deepcopy(value.get("canonical_spo", {})),
            query=QueryStructureProfile.from_dict(value.get("query")),
            schema=SchemaProfile.from_dict(value.get("schema")),
            temporal=TemporalProfile.from_dict(value.get("temporal")),
            masked_action_space=tuple(value.get("masked_action_space", [])),
            active_profile_fields=tuple(value.get("active_profile_fields", [])),
        )


@dataclass
class MemoryStatistics:
    """Validation and reuse counters attached to one memory record."""

    ns: int = 0
    nf: int = 0
    nr: int = 0

    def __post_init__(self) -> None:
        if min(self.ns, self.nf, self.nr) < 0:
            raise ValueError("Memory statistics must be non-negative")

    def record_validation(self, success: bool, count: int = 1) -> None:
        if count < 1:
            raise ValueError("Validation count must be positive")
        if success:
            self.ns += count
        else:
            self.nf += count

    def record_reuse(self, count: int = 1) -> None:
        if count < 1:
            raise ValueError("Reuse count must be positive")
        self.nr += count

    def merge(self, other: "MemoryStatistics") -> None:
        self.ns += other.ns
        self.nf += other.nf
        self.nr += other.nr

    def confidence(self, alpha: float = 1.0, beta: float = 1.0) -> float:
        """Return the beta-smoothed reliability estimate used by retrieval."""

        denominator = self.ns + self.nf + alpha + beta
        return (self.ns + alpha) / denominator

    def to_dict(self) -> dict[str, int]:
        return {"Ns": self.ns, "Nf": self.nf, "Nr": self.nr}

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "MemoryStatistics":
        value = value or {}
        return cls(
            ns=int(value.get("Ns", value.get("ns", 0))),
            nf=int(value.get("Nf", value.get("nf", 0))),
            nr=int(value.get("Nr", value.get("nr", 0))),
        )


@dataclass
class ValidatedStateTransition:
    """Reusable action and parameterized state delta produced by one skill."""

    selected_action: Any
    state_delta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_action": deepcopy(self.selected_action),
            "state_delta": deepcopy(self.state_delta),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ValidatedStateTransition":
        return cls(
            selected_action=deepcopy(value.get("selected_action")),
            state_delta=deepcopy(value.get("state_delta", {})),
        )


@dataclass
class StateMemoryRecord:
    """One reusable, profile-scoped state transition."""

    index: MemoryIndex
    profile: StateMemoryProfile
    transition: ValidatedStateTransition
    statistics: MemoryStatistics = field(default_factory=MemoryStatistics)
    metadata: dict[str, Any] = field(default_factory=dict)
    record_id: str = ""

    def __post_init__(self) -> None:
        self.record_id = str(self.record_id).strip() or stable_fingerprint(
            {
                "index": self.index.to_dict(),
                "profile_key": self.profile.profile_key,
                "transition": self.transition.to_dict(),
            }
        )

    @property
    def profile_key(self) -> str:
        return self.profile.profile_key

    @property
    def record_key(self) -> str:
        """Backward-compatible alias for the stable record identifier."""

        return self.record_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "record_key": self.record_key,
            "profile_key": self.profile_key,
            "index": self.index.to_dict(),
            "profile": self.profile.to_dict(),
            "transition": self.transition.to_dict(),
            "statistics": self.statistics.to_dict(),
            "metadata": deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StateMemoryRecord":
        return cls(
            index=MemoryIndex.from_dict(value["index"]),
            profile=StateMemoryProfile.from_dict(value["profile"]),
            transition=ValidatedStateTransition.from_dict(value["transition"]),
            statistics=MemoryStatistics.from_dict(value.get("statistics")),
            metadata=deepcopy(value.get("metadata", {})),
            record_id=str(value.get("record_id") or value.get("record_key") or ""),
        )


def _canonical_value(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _canonical_value(value.to_dict())
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, set):
        canonical_items = [_canonical_value(item) for item in value]
        return sorted(canonical_items, key=canonical_json)
    return value
