"""Compatibility-gated profile matching for TSSDA state memory."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import random
from typing import Any, Callable, Iterable

from .models import (
    NONE,
    NOT_APPLICABLE,
    UNKNOWN,
    UNSET,
    MemoryIndex,
    StateMemoryProfile,
    StateMemoryRecord,
    canonical_json,
    temporal_signature_similarity,
)


ScoreFunction = Callable[[Any, Any], float]
MISSING_PROFILE_VALUES = {None, "", UNSET, UNKNOWN}


@dataclass(frozen=True)
class MemoryMatchConfig:
    """Thresholds for a reliable and sufficiently evidenced memory hit."""

    min_profile_similarity: float = 0.80
    min_evidence_coverage: float = 0.75
    min_reliability: float = 0.60
    min_validation_successes: int = 1
    reliability_alpha: float = 1.0
    reliability_beta: float = 1.0
    reuse_penalty: float = 0.05
    deletion_confidence_threshold: float = 0.35
    temporal_failure_similarity_threshold: float = 0.95
    temporal_failure_min_count: int = 3
    top_k: int = 5
    random_seed: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "min_profile_similarity",
            "min_evidence_coverage",
            "min_reliability",
            "deletion_confidence_threshold",
            "temporal_failure_similarity_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.min_validation_successes < 0:
            raise ValueError("min_validation_successes must be non-negative")
        if self.reliability_alpha <= 0 or self.reliability_beta <= 0:
            raise ValueError("Reliability smoothing parameters must be positive")
        if self.reuse_penalty < 0:
            raise ValueError("reuse_penalty must be non-negative")
        if self.temporal_failure_min_count < 1:
            raise ValueError("temporal_failure_min_count must be positive")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")


@dataclass(frozen=True)
class MemoryMatch:
    """One compatible memory record with explainable matching scores."""

    record: StateMemoryRecord
    profile_similarity: float
    evidence_coverage: float
    reliability: float
    rank_score: float
    selection_probability: float = 0.0
    group_scores: dict[str, float] = field(default_factory=dict)
    feature_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record.record_id,
            "record_key": self.record.record_key,
            "index": self.record.index.to_dict(),
            "selected_action": self.record.transition.selected_action,
            "profile_similarity": self.profile_similarity,
            "evidence_coverage": self.evidence_coverage,
            "reliability": self.reliability,
            "rank_score": self.rank_score,
            "selection_probability": self.selection_probability,
            "group_scores": dict(self.group_scores),
            "feature_scores": dict(self.feature_scores),
        }


@dataclass(frozen=True)
class MemoryRetrievalResult:
    """Result of one strict-index lookup followed by profile matching."""

    index: MemoryIndex
    candidate_count: int
    matches: tuple[MemoryMatch, ...] = ()
    selected_match: MemoryMatch | None = None
    rejections: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def hit(self) -> bool:
        return self.selected_match is not None

    @property
    def best_match(self) -> MemoryMatch | None:
        return self.matches[0] if self.matches else None


@dataclass(frozen=True)
class _FeatureRule:
    path: str
    group: str
    weight: float
    score: ScoreFunction


@dataclass(frozen=True)
class _ProfileScore:
    similarity: float
    coverage: float
    group_scores: dict[str, float]
    feature_scores: dict[str, float]


class StateMemoryMatcher:
    """Retrieve valid transitions using task-aware profile similarity."""

    def __init__(self, config: MemoryMatchConfig | None = None):
        self.config = config or MemoryMatchConfig()
        self._random = random.Random(self.config.random_seed)

    def retrieve(
        self,
        *,
        index: MemoryIndex,
        current_profile: StateMemoryProfile,
        candidates: Iterable[StateMemoryRecord],
        top_k: int | None = None,
    ) -> MemoryRetrievalResult:
        records = list(candidates)
        matches: list[MemoryMatch] = []
        rejections: dict[str, tuple[str, ...]] = {}

        for record in records:
            reasons = self._hard_rejection_reasons(
                index=index,
                current=current_profile,
                record=record,
            )
            reliability = record.statistics.confidence(
                alpha=self.config.reliability_alpha,
                beta=self.config.reliability_beta,
            )
            if record.statistics.ns < self.config.min_validation_successes:
                reasons.append("insufficient_validation_successes")
            if reliability < self.config.min_reliability:
                reasons.append("reliability_below_threshold")

            profile_score = _score_profiles(current_profile, record.profile)
            if profile_score.coverage < self.config.min_evidence_coverage:
                reasons.append("evidence_coverage_below_threshold")
            if profile_score.similarity < self.config.min_profile_similarity:
                reasons.append("profile_similarity_below_threshold")

            if reasons:
                rejections[record.record_key] = tuple(dict.fromkeys(reasons))
                continue

            rank_score = (
                profile_score.similarity
                * reliability
                / (1.0 + self.config.reuse_penalty * record.statistics.nr)
            )
            matches.append(
                MemoryMatch(
                    record=StateMemoryRecord.from_dict(record.to_dict()),
                    profile_similarity=_rounded(profile_score.similarity),
                    evidence_coverage=_rounded(profile_score.coverage),
                    reliability=_rounded(reliability),
                    rank_score=_rounded(rank_score),
                    group_scores=profile_score.group_scores,
                    feature_scores=profile_score.feature_scores,
                )
            )

        matches.sort(
            key=lambda match: (
                -match.rank_score,
                -match.profile_similarity,
                -match.reliability,
                match.record.statistics.nr,
                match.record.record_key,
            )
        )
        limit = self.config.top_k if top_k is None else top_k
        if limit < 1:
            raise ValueError("top_k must be positive")
        eligible_matches = _attach_selection_probabilities(matches[:limit])
        selected_match = self._sample_match(eligible_matches)
        return MemoryRetrievalResult(
            index=index,
            candidate_count=len(records),
            matches=eligible_matches,
            selected_match=selected_match,
            rejections=rejections,
        )

    def _sample_match(
        self,
        matches: tuple[MemoryMatch, ...],
    ) -> MemoryMatch | None:
        """Sample one hit using normalized rank scores."""

        if not matches:
            return None
        draw = self._random.random()
        cumulative = 0.0
        for match in matches:
            cumulative += match.selection_probability
            if draw <= cumulative:
                return match
        return matches[-1]

    def _hard_rejection_reasons(
        self,
        *,
        index: MemoryIndex,
        current: StateMemoryProfile,
        record: StateMemoryRecord,
    ) -> list[str]:
        remembered = record.profile
        reasons: list[str] = []
        reasons.extend(temporal_rejection_reasons(current.temporal, remembered.temporal))
        if record.index != index:
            reasons.append("index_mismatch")
        if remembered.profile_version != current.profile_version:
            reasons.append("profile_version_mismatch")
        if canonical_json(remembered.canonical_spo) != canonical_json(
            current.canonical_spo
        ):
            reasons.append("canonical_spo_mismatch")

        current_actions = set(current.masked_action_space)
        remembered_actions = set(remembered.masked_action_space)
        selected_action = _selected_action_name(record.transition.selected_action)
        if current_actions:
            if selected_action is None:
                reasons.append("selected_action_not_identifiable")
            elif selected_action not in current_actions:
                reasons.append("selected_action_masked")
        if remembered_actions and selected_action not in remembered_actions:
            reasons.append("corrupt_remembered_action_space")

        if _field_is_active(current, "query.select_shape") and _incompatible_select_shapes(
            current.query.select_shape,
            remembered.query.select_shape,
        ):
            reasons.append("select_shape_incompatible")
        if _field_is_active(current, "query.select_type_family") and _disjoint_type_families(
            current.query.select_type_family,
            remembered.query.select_type_family,
        ):
            reasons.append("data_type_family_incompatible")
        if _field_is_active(current, "schema.data_type_family") and _disjoint_type_families(
            current.schema.data_type_family,
            remembered.schema.data_type_family,
        ):
            reasons.append("data_type_family_incompatible")

        for field_name in (
            "device_count",
            "measurement_name_count",
            "timeseries_count",
        ):
            if _field_is_active(current, f"schema.{field_name}") and _zero_nonzero_mismatch(
                getattr(current.schema, field_name),
                getattr(remembered.schema, field_name),
            ):
                reasons.append(f"schema_{field_name}_incompatible")

        if _field_is_active(current, "temporal.data_span") and _empty_nonempty_mismatch(
            current.temporal.data_span,
            remembered.temporal.data_span,
        ):
            reasons.append("temporal_data_availability_incompatible")
        if _field_is_active(current, "temporal.query_scope.data_overlap") and _temporal_applicability_mismatch(
            current.temporal.query_scope.data_overlap,
            remembered.temporal.query_scope.data_overlap,
        ):
            reasons.append("temporal_query_scope_incompatible")
        if _field_is_active(current, "temporal.window_shape.step_relation") and _temporal_applicability_mismatch(
            current.temporal.window_shape.step_relation,
            remembered.temporal.window_shape.step_relation,
        ):
            reasons.append("window_applicability_incompatible")
        if _matches_failed_temporal_context(
            current,
            remembered,
            min_failures=self.config.temporal_failure_min_count,
            similarity_threshold=(
                self.config.temporal_failure_similarity_threshold
            ),
        ):
            reasons.append("learned_temporal_context_incompatible")

        if (
            selected_action == "extend_binary_series"
            and _field_is_active(current, "temporal.coverage_overlap")
            and current.temporal.coverage_overlap == "disjoint"
        ):
            reasons.append("binary_series_has_disjoint_temporal_coverage")
        return reasons


def _score_profiles(
    current: StateMemoryProfile,
    remembered: StateMemoryProfile,
) -> _ProfileScore:
    weighted_score = 0.0
    current_weight = 0.0
    comparable_weight = 0.0
    group_scores: dict[str, float] = {}
    group_weights: dict[str, float] = {}
    feature_scores: dict[str, float] = {}

    for rule in _FEATURE_RULES:
        if not _field_is_active(current, rule.path):
            continue
        current_value = _profile_value(current, rule.path)
        if _is_missing(current_value):
            continue
        current_weight += rule.weight
        group_weights[rule.group] = group_weights.get(rule.group, 0.0) + rule.weight

        remembered_value = _profile_value(remembered, rule.path)
        if _is_missing(remembered_value):
            feature_scores[rule.path] = 0.0
            continue
        comparable_weight += rule.weight
        score = max(0.0, min(1.0, rule.score(current_value, remembered_value)))
        feature_scores[rule.path] = _rounded(score)
        contribution = rule.weight * score
        weighted_score += contribution
        group_scores[rule.group] = group_scores.get(rule.group, 0.0) + contribution

    normalized_groups = {
        group: _rounded(group_scores.get(group, 0.0) / weight)
        for group, weight in group_weights.items()
        if weight > 0
    }
    if current_weight <= 0:
        return _ProfileScore(0.0, 0.0, normalized_groups, feature_scores)
    return _ProfileScore(
        similarity=weighted_score / current_weight,
        coverage=comparable_weight / current_weight,
        group_scores=normalized_groups,
        feature_scores=feature_scores,
    )


def temporal_rejection_reasons(current, remembered) -> list[str]:
    """Known sampling incompatibilities cannot be diluted by other features."""
    reasons = []
    for name in ("timestamp_alignment", "sampling_regularity", "point_distribution"):
        left, right = getattr(current, name), getattr(remembered, name)
        if not _is_missing(left) and not _is_missing(right) and left != right:
            reasons.append(f"temporal_{name}_incompatible")
    order = ("subsecond", "second", "minute", "hour", "day_or_more")
    left, right = current.sampling_interval, remembered.sampling_interval
    if left in order and right in order and abs(order.index(left) - order.index(right)) > 1:
        reasons.append("temporal_sampling_interval_incompatible")
    return reasons


def score_profiles(current, remembered):
    """Shared feature-wise Sim and evidence coverage for both memory levels."""
    return _score_profiles(current, remembered)


def _profile_value(profile: StateMemoryProfile, path: str) -> Any:
    if path == "masked_action_space":
        return profile.masked_action_space
    value: Any = profile
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _matches_failed_temporal_context(
    current: StateMemoryProfile,
    remembered: StateMemoryProfile,
    *,
    min_failures: int,
    similarity_threshold: float,
) -> bool:
    """Return true when the current context hits a mature negative boundary."""

    for failure_context in remembered.temporal.failure_contexts:
        if failure_context.failures < min_failures:
            continue
        current_signature = {}
        for path in failure_context.signature:
            if not _field_is_active(current, path):
                current_signature = {}
                break
            current_value = _profile_value(current, path)
            if _is_missing(current_value):
                current_signature = {}
                break
            current_signature[path] = current_value
        if current_signature and temporal_signature_similarity(
            current_signature,
            failure_context.signature,
        ) >= similarity_threshold:
            return True
    return False


def _field_is_active(profile: StateMemoryProfile, path: str) -> bool:
    fields = profile.active_profile_fields
    return not fields or path in fields


def _is_missing(value: Any) -> bool:
    if isinstance(value, (tuple, list, set)):
        return not value
    try:
        return value in MISSING_PROFILE_VALUES
    except TypeError:
        return False


def _selected_action_name(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ("selected_action", "decision", "action"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _exact(current: Any, remembered: Any) -> float:
    return 1.0 if current == remembered else 0.0


def _action_set_similarity(current: Any, remembered: Any) -> float:
    left, right = set(current), set(remembered)
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _ordered_similarity(order: tuple[str, ...], floor: float = 0.1) -> ScoreFunction:
    positions = {value: index for index, value in enumerate(order)}

    def score(current: Any, remembered: Any) -> float:
        if current == remembered:
            return 1.0
        if current not in positions or remembered not in positions:
            return 0.0
        distance = abs(positions[current] - positions[remembered])
        return max(floor, 1.0 - 0.3 * distance)

    return score


def _cardinality_similarity(current: Any, remembered: Any) -> float:
    if current == remembered:
        return 1.0
    if "zero" in {current, remembered}:
        return 0.0
    return _ordered_similarity(("one", "few", "many"), floor=0.25)(
        current,
        remembered,
    )


def _type_similarity(current: Any, remembered: Any) -> float:
    left = _type_family_set(current)
    right = _type_family_set(remembered)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _select_shape_similarity(current: Any, remembered: Any) -> float:
    if current == remembered:
        return 1.0
    if "mixed" in {current, remembered}:
        return 0.5
    return 0.0


def _sharing_similarity(current: Any, remembered: Any) -> float:
    if current == remembered:
        return 1.0
    pairs = {current, remembered}
    if pairs == {"full", "partial"}:
        return 0.65
    if pairs == {"partial", "none"}:
        return 0.35
    return 0.0


def _coverage_similarity(current: Any, remembered: Any) -> float:
    if current == remembered:
        return 1.0
    pairs = {current, remembered}
    if pairs == {"single", "common"}:
        return 0.8
    if pairs in ({"common", "partial"}, {"single", "partial"}):
        return 0.6
    if pairs == {"partial", "disjoint"}:
        return 0.25
    return 0.0


def _overlap_similarity(current: Any, remembered: Any) -> float:
    if current == remembered:
        return 1.0
    pairs = {current, remembered}
    if pairs == {"all", "some"}:
        return 0.65
    if pairs == {"some", "none"}:
        return 0.2
    return 0.0


def _position_similarity(current: Any, remembered: Any) -> float:
    if current == remembered:
        return 1.0
    pairs = {current, remembered}
    if pairs in ({"head", "middle"}, {"middle", "tail"}):
        return 0.55
    if "full" in pairs and not pairs & {"outside"}:
        return 0.4
    return 0.0


def _step_similarity(current: Any, remembered: Any) -> float:
    if current == remembered:
        return 1.0
    if {current, remembered} == {"tumbling", "overlapping"}:
        return 0.55
    return 0.1 if "gapped" in {current, remembered} else 0.0


def _clause_mask_similarity(current: Any, remembered: Any) -> float:
    left, right = int(current), int(remembered)
    union = left | right
    if union == 0:
        return 1.0
    return (left & right).bit_count() / union.bit_count()


def _intent_similarity(current: Any, remembered: Any) -> float:
    return 1.0 if current == remembered else 0.15


def _type_family_set(value: Any) -> set[str]:
    mapping = {
        "numeric_only": {"numeric"},
        "text_only": {"text"},
        "boolean_only": {"boolean"},
        "mixed": {"numeric", "text", "boolean"},
    }
    return mapping.get(str(value), set())


def _disjoint_type_families(current: Any, remembered: Any) -> bool:
    if _is_missing(current) or _is_missing(remembered):
        return False
    left, right = _type_family_set(current), _type_family_set(remembered)
    return bool(left and right and left.isdisjoint(right))


def _incompatible_select_shapes(current: Any, remembered: Any) -> bool:
    if _is_missing(current) or _is_missing(remembered) or current == remembered:
        return False
    return "mixed" not in {current, remembered}


def _zero_nonzero_mismatch(current: Any, remembered: Any) -> bool:
    if _is_missing(current) or _is_missing(remembered):
        return False
    return (current == "zero") != (remembered == "zero")


def _empty_nonempty_mismatch(current: Any, remembered: Any) -> bool:
    if _is_missing(current) or _is_missing(remembered):
        return False
    return (current == "empty") != (remembered == "empty")


def _temporal_applicability_mismatch(current: Any, remembered: Any) -> bool:
    if _is_missing(current) or _is_missing(remembered):
        return False
    inactive = {NONE, NOT_APPLICABLE}
    return (current in inactive) != (remembered in inactive)


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _attach_selection_probabilities(
    matches: list[MemoryMatch],
) -> tuple[MemoryMatch, ...]:
    """Normalize Rank(m) into P(m | state) over eligible candidates."""

    if not matches:
        return ()
    score_sum = sum(max(match.rank_score, 0.0) for match in matches)
    if score_sum <= 0:
        probability = 1.0 / len(matches)
        return tuple(
            replace(match, selection_probability=probability)
            for match in matches
        )
    probabilities = [max(match.rank_score, 0.0) / score_sum for match in matches]
    # Assign residual floating-point mass to the last candidate.
    probabilities[-1] = max(0.0, 1.0 - sum(probabilities[:-1]))
    return tuple(
        replace(match, selection_probability=probability)
        for match, probability in zip(matches, probabilities)
    )


_FEATURE_RULES = (
    _FeatureRule("masked_action_space", "action_space", 0.08, _action_set_similarity),
    _FeatureRule("query.select_shape", "query", 0.08, _select_shape_similarity),
    _FeatureRule(
        "query.select_layout.group_count",
        "query",
        0.025,
        _cardinality_similarity,
    ),
    _FeatureRule(
        "query.select_layout.expressions_per_group",
        "query",
        0.035,
        _cardinality_similarity,
    ),
    _FeatureRule("query.select_type_family", "query", 0.06, _type_similarity),
    _FeatureRule(
        "query.expression_depth",
        "query",
        0.04,
        _ordered_similarity(("atomic", "shallow", "nested", "deep")),
    ),
    _FeatureRule("query.clause_mask", "query", 0.06, _clause_mask_similarity),
    _FeatureRule("schema.device_count", "schema", 0.03, _cardinality_similarity),
    _FeatureRule(
        "schema.measurement_name_count",
        "schema",
        0.035,
        _cardinality_similarity,
    ),
    _FeatureRule(
        "schema.timeseries_count",
        "schema",
        0.045,
        _cardinality_similarity,
    ),
    _FeatureRule(
        "schema.measurement_sharing",
        "schema",
        0.05,
        _sharing_similarity,
    ),
    _FeatureRule("schema.data_type_family", "schema", 0.08, _type_similarity),
    _FeatureRule(
        "temporal.intent_pattern",
        "temporal",
        0.08,
        _intent_similarity,
    ),
    _FeatureRule(
        "temporal.data_span",
        "temporal",
        0.04,
        _ordered_similarity(("point", "short", "medium", "long")),
    ),
    _FeatureRule(
        "temporal.coverage_overlap",
        "temporal",
        0.06,
        _coverage_similarity,
    ),
    _FeatureRule("temporal.sampling_interval", "temporal", .05,
                 _ordered_similarity(("subsecond", "second", "minute", "hour", "day_or_more"))),
    _FeatureRule("temporal.sampling_regularity", "temporal", .04, _exact),
    _FeatureRule("temporal.timestamp_alignment", "temporal", .08, _exact),
    _FeatureRule("temporal.point_distribution", "temporal", .05, _exact),
    _FeatureRule(
        "temporal.query_scope.data_overlap",
        "temporal",
        0.06,
        _overlap_similarity,
    ),
    _FeatureRule(
        "temporal.query_scope.span_ratio",
        "temporal",
        0.04,
        _ordered_similarity(("narrow", "medium", "broad", "oversized")),
    ),
    _FeatureRule(
        "temporal.query_scope.position",
        "temporal",
        0.035,
        _position_similarity,
    ),
    _FeatureRule(
        "temporal.window_shape.window_count",
        "temporal",
        0.025,
        _cardinality_similarity,
    ),
    _FeatureRule(
        "temporal.window_shape.step_relation",
        "temporal",
        0.04,
        _step_similarity,
    ),
)
