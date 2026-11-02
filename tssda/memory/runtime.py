"""Runtime orchestration for state-memory warmup, reuse, and admission."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable

from .matcher import (
    MemoryMatch,
    MemoryMatchConfig,
    MemoryRetrievalResult,
    StateMemoryMatcher,
)
from .models import (
    MemoryIndex,
    MemoryStatistics,
    StateMemoryProfile,
    StateMemoryRecord,
    TemporalProfile,
    UNKNOWN,
    UNSET,
    canonical_json,
    stable_fingerprint,
)
from .profiles import StateProfileBuilder
from .expressions import ExpressionMemoryController
from .store import StateMemoryStore
from ..token_accounting import (
    empty_memory_token_summary,
    estimate_json_tokens,
    summarize_memory_token_events,
)


STATE_MEMORY_PHASES = frozenset({"P2", "P3", "P6", "P7", "P8", "P9"})


@dataclass(frozen=True)
class MemoryActivationPolicy:
    """Require a diverse validated corpus before decisions may be reused."""

    min_index_records: int = 6
    min_distinct_profiles: int = 3
    min_distinct_actions: int = 2
    min_eligible_matches: int = 1

    def __post_init__(self) -> None:
        for name in (
            "min_index_records",
            "min_distinct_profiles",
            "min_distinct_actions",
            "min_eligible_matches",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class MemoryActivationStatus:
    active: bool
    validated_record_count: int
    distinct_profile_count: int
    distinct_action_count: int
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryValidationResult:
    """Outcome of the final SQL validator that gates long-term admission."""

    validated: bool
    passed: bool
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.passed and not self.validated:
            raise ValueError("A passed validation result must be validated")

    def to_dict(self) -> dict[str, Any]:
        return {
            "validated": self.validated,
            "passed": self.passed,
            "reason": self.reason,
            "details": deepcopy(self.details),
        }

    @classmethod
    def passed_result(
        cls,
        reason: str = "SQL validation passed.",
        **details: Any,
    ) -> "MemoryValidationResult":
        return cls(True, True, reason, details)

    @classmethod
    def failed_result(
        cls,
        reason: str = "SQL validation failed.",
        **details: Any,
    ) -> "MemoryValidationResult":
        return cls(True, False, reason, details)

    @classmethod
    def not_run(
        cls,
        reason: str = "SQL validation has not been configured.",
        **details: Any,
    ) -> "MemoryValidationResult":
        return cls(False, False, reason, details)


@dataclass
class MemoryLookup:
    """One decision-time memory lookup, including warmup diagnostics."""

    index: MemoryIndex
    profile: StateMemoryProfile
    activation: MemoryActivationStatus
    read_enabled: bool = True
    retrieval: MemoryRetrievalResult | None = None

    @property
    def selected_match(self) -> MemoryMatch | None:
        if self.retrieval is None:
            return None
        return self.retrieval.selected_match

    @property
    def hit(self) -> bool:
        return self.selected_match is not None

    def to_trace_dict(self) -> dict[str, Any]:
        selected = self.selected_match
        return {
            "enabled": True,
            "read_enabled": self.read_enabled,
            "warm": self.activation.active,
            "hit": self.hit,
            "index": self.index.to_dict(),
            "profile_key": self.profile.profile_key,
            "activation": {
                "validated_record_count": self.activation.validated_record_count,
                "distinct_profile_count": self.activation.distinct_profile_count,
                "distinct_action_count": self.activation.distinct_action_count,
                "reasons": list(self.activation.reasons),
            },
            "candidate_count": (
                self.retrieval.candidate_count if self.retrieval else 0
            ),
            "eligible_match_count": (
                len(self.retrieval.matches) if self.retrieval else 0
            ),
            "selected_record_key": (
                selected.record.record_key if selected else None
            ),
            "selected_record_id": (
                selected.record.record_id if selected else None
            ),
            "selection_probability": (
                selected.selection_probability if selected else 0.0
            ),
            "rank_score": selected.rank_score if selected else 0.0,
        }


@dataclass
class _PendingDecision:
    lookup: MemoryLookup
    selected_action: str
    state_delta: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MemoryReuseTrace:
    """Trajectory-local evidence captured for one adopted memory record."""

    record_id: str
    index: MemoryIndex
    remembered_temporal_profile: TemporalProfile
    current_temporal_profile: TemporalProfile
    active_temporal_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "index": self.index.to_dict(),
            "remembered_temporal_profile": (
                self.remembered_temporal_profile.to_dict()
            ),
            "current_temporal_profile": self.current_temporal_profile.to_dict(),
            "active_temporal_fields": list(self.active_temporal_fields),
        }


class StateMemoryController:
    """Manage state-memory reads and delayed trajectory-level writes."""

    def __init__(
        self,
        store: StateMemoryStore,
        *,
        match_config: MemoryMatchConfig | None = None,
        activation_policy: MemoryActivationPolicy | None = None,
        read_enabled: bool = True,
        autosave: bool = True,
    ):
        self.store = store
        self.match_config = match_config or MemoryMatchConfig()
        self.activation_policy = activation_policy or MemoryActivationPolicy()
        self.read_enabled = bool(read_enabled)
        self.autosave = autosave
        self.profile_builder = StateProfileBuilder()
        self.matcher = StateMemoryMatcher(self.match_config)
        self.expression_memory = ExpressionMemoryController(
            store.expressions, self.match_config, read_enabled=self.read_enabled,
        )
        self._pending: dict[str, _PendingDecision] = {}
        self._reuse_buffer: list[MemoryReuseTrace] = []
        self._lookup_cache: dict[str, MemoryLookup] = {}
        self._token_accounting_events: list[dict[str, Any]] = []
        self.last_token_accounting_events: tuple[dict[str, Any], ...] = ()
        self.last_token_accounting_summary = empty_memory_token_summary()
        self._temporal_prefetch_attempted = False
        self.last_deleted_record_ids: tuple[str, ...] = ()
        self.last_validation_result = MemoryValidationResult.not_run(
            "No trajectory has been finalized yet."
        )

    @property
    def provisional_count(self) -> int:
        """Number of fresh agent decisions waiting for SQL validation."""

        return len(self._pending)

    @property
    def reused_count(self) -> int:
        """Number of memory hits recorded by the current trajectory."""

        return len(self._reuse_buffer)

    @property
    def reuse_buffer(self) -> tuple[dict[str, Any], ...]:
        """Return read-only snapshots of the current temporal reuse evidence."""

        return tuple(trace.to_dict() for trace in self._reuse_buffer)

    def begin_trajectory(self) -> None:
        self.expression_memory.begin_trajectory()
        self._pending.clear()
        self._reuse_buffer.clear()
        self._lookup_cache.clear()
        self._token_accounting_events.clear()
        self.last_token_accounting_events = ()
        self.last_token_accounting_summary = empty_memory_token_summary()
        self._temporal_prefetch_attempted = False
        self.last_deleted_record_ids = ()
        self.last_validation_result = MemoryValidationResult.not_run(
            "Current trajectory has not reached final SQL validation."
        )

    def supports(self, phase_id: str, skill: Any) -> bool:
        return (
            phase_id in STATE_MEMORY_PHASES
            and bool(getattr(skill, "memory_enabled", False))
        )

    def should_prefetch_temporal_profile(
        self,
        *,
        state: Any,
        phase_id: str,
        skill: Any,
    ) -> bool:
        """Collect bounded evidence once at P2 for both memory levels."""

        if (
            not self.read_enabled
            or self._temporal_prefetch_attempted
            or phase_id != "P2"
        ):
            return False
        fields = set(getattr(skill, "memory_profile_fields", ()))
        if not fields & {"temporal.data_span", "temporal.coverage_overlap"}:
            return False
        context = getattr(state, "memory_context", None)
        if context is not None and getattr(context, "temporal_observations", None):
            return False
        self._temporal_prefetch_attempted = True
        return True

    def lookup(
        self,
        *,
        state: Any,
        phase_id: str,
        skill: Any,
        action_space: Iterable[str],
    ) -> MemoryLookup:
        actions = tuple(sorted({str(action) for action in action_space if action}))
        index = self.profile_builder.build_index(
            state=state,
            phase_id=phase_id,
            skill_id=skill.skill_id,
        )
        profile = self.profile_builder.build_profile(
            state=state,
            required_spo=getattr(skill, "memory_required_spo", ()),
            masked_action_space=actions,
            active_profile_fields=getattr(skill, "memory_profile_fields", ()),
        )
        cache_key = f"{index.phase_id}:{index.skill_id}:{index.intent}:{profile.profile_key}"
        cached = self._lookup_cache.get(cache_key)
        if cached is not None:
            return cached

        if not self.read_enabled:
            lookup = MemoryLookup(
                index=index,
                profile=profile,
                activation=MemoryActivationStatus(
                    active=False,
                    validated_record_count=0,
                    distinct_profile_count=0,
                    distinct_action_count=0,
                    reasons=("memory_reads_disabled",),
                ),
                read_enabled=False,
                retrieval=None,
            )
            self._lookup_cache[cache_key] = lookup
            return lookup

        records = self.store.candidates(
            phase_id=index.phase_id,
            skill_id=index.skill_id,
            intent=index.intent,
        )
        activation = self._activation_status(records, actions)
        retrieval = None
        if activation.active:
            retrieval = self.matcher.retrieve(
                index=index,
                current_profile=profile,
                candidates=records,
            )
            if len(retrieval.matches) < self.activation_policy.min_eligible_matches:
                retrieval = MemoryRetrievalResult(
                    index=index,
                    candidate_count=retrieval.candidate_count,
                    matches=(),
                    selected_match=None,
                    rejections=deepcopy(retrieval.rejections),
                )
        lookup = MemoryLookup(
            index=index,
            profile=profile,
            activation=activation,
            read_enabled=True,
            retrieval=retrieval,
        )
        self._lookup_cache[cache_key] = lookup
        return lookup

    def stage_decision(
        self,
        *,
        lookup: MemoryLookup,
        selected_action: str,
        state_delta: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if lookup.hit:
            return
        pending = _PendingDecision(
            lookup=lookup,
            selected_action=selected_action,
            state_delta=deepcopy(state_delta),
            metadata=deepcopy(metadata or {}),
        )
        pending_key = stable_fingerprint(
            {
                "index": lookup.index.to_dict(),
                "profile_key": lookup.profile.profile_key,
            }
        )
        self._pending[pending_key] = pending

    def record_reuse_attempt(
        self,
        match: MemoryMatch,
        *,
        current_profile: StateMemoryProfile,
    ) -> None:
        """Capture temporal evidence for every adopted memory decision."""

        self._reuse_buffer.append(
            MemoryReuseTrace(
                record_id=match.record.record_id,
                index=match.record.index,
                remembered_temporal_profile=TemporalProfile.from_dict(
                    match.record.profile.temporal.to_dict()
                ),
                current_temporal_profile=TemporalProfile.from_dict(
                    current_profile.temporal.to_dict()
                ),
                active_temporal_fields=tuple(
                    field
                    for field in current_profile.active_profile_fields
                    if field.startswith("temporal.")
                ),
            )
        )

    def record_token_accounting(self, event: dict[str, Any]) -> None:
        """Record one retry-aware Memory lookup and its prompt-token effect."""

        self._token_accounting_events.append(deepcopy(event))

    def _settle_token_accounting(self) -> None:
        self.last_token_accounting_events = tuple(
            deepcopy(self._token_accounting_events)
        )
        self.last_token_accounting_summary = summarize_memory_token_events(
            self._token_accounting_events
        )
        self._token_accounting_events.clear()

    def finish_trajectory(
        self,
        *,
        validation_result: MemoryValidationResult,
    ) -> None:
        """Commit or discard the provisional buffer after explicit validation."""

        self.last_validation_result = validation_result
        self._settle_token_accounting()
        changed = self.expression_memory.finish_trajectory(validation_result)
        failed_records: dict[tuple[MemoryIndex, str], None] = {}
        for reuse_trace in self._reuse_buffer:
            # Nr measures actual reuse, independently of whether the external
            # SQL validator has been configured or completed.
            self.store.record_reuse(
                index=reuse_trace.index,
                record_id=reuse_trace.record_id,
            )
            changed = True
            if validation_result.validated:
                self.store.record_validation(
                    index=reuse_trace.index,
                    record_id=reuse_trace.record_id,
                    success=validation_result.passed,
                )
            if validation_result.validated and not validation_result.passed:
                failed_records[(reuse_trace.index, reuse_trace.record_id)] = None
                failure_signature = _temporal_failure_signature(reuse_trace)
                if failure_signature:
                    self.store.refine_temporal_profile(
                        index=reuse_trace.index,
                        record_id=reuse_trace.record_id,
                        failure_signature=failure_signature,
                        similarity_threshold=(
                            self.match_config.temporal_failure_similarity_threshold
                        ),
                    )

        deleted_record_ids = []
        for index, record_id in failed_records:
            record = self.store.get(index=index, record_id=record_id)
            confidence = record.statistics.confidence(
                alpha=self.match_config.reliability_alpha,
                beta=self.match_config.reliability_beta,
            )
            if confidence <= self.match_config.deletion_confidence_threshold:
                if self.store.remove(index=index, record_id=record_id):
                    deleted_record_ids.append(record_id)
                    changed = True
        self.last_deleted_record_ids = tuple(sorted(deleted_record_ids))

        if validation_result.passed:
            for pending in self._pending.values():
                metadata = deepcopy(pending.metadata)
                metadata["admission_validation"] = {
                    "reason": validation_result.reason,
                    "details": deepcopy(validation_result.details),
                }
                self.store.admit(
                    index=pending.lookup.index,
                    profile=pending.lookup.profile,
                    selected_action=pending.selected_action,
                    state_delta=pending.state_delta,
                    statistics=MemoryStatistics(ns=1),
                    metadata=metadata,
                )
                changed = True

        self._pending.clear()
        self._reuse_buffer.clear()
        self._lookup_cache.clear()
        self._temporal_prefetch_attempted = False
        if changed and self.autosave:
            self.store.save()

    def discard_trajectory(
        self,
        *,
        validation_result: MemoryValidationResult,
    ) -> None:
        """Drop provisional state without changing persistent Memory."""

        self.last_validation_result = validation_result
        self._settle_token_accounting()
        self.expression_memory.begin_trajectory()
        self.last_deleted_record_ids = ()
        self._pending.clear()
        self._reuse_buffer.clear()
        self._lookup_cache.clear()
        self._temporal_prefetch_attempted = False

    def _activation_status(
        self,
        records: Iterable[StateMemoryRecord],
        action_space: tuple[str, ...],
    ) -> MemoryActivationStatus:
        validated = [
            record
            for record in records
            if record.statistics.ns >= self.match_config.min_validation_successes
            and record.statistics.confidence(
                alpha=self.match_config.reliability_alpha,
                beta=self.match_config.reliability_beta,
            )
            >= self.match_config.min_reliability
        ]
        profiles = {record.profile_key for record in validated}
        actions = {
            action
            for record in validated
            if (action := _selected_action(record.transition.selected_action))
            and action in action_space
        }
        required_actions = min(
            self.activation_policy.min_distinct_actions,
            max(1, len(action_space)),
        )
        reasons = []
        if len(validated) < self.activation_policy.min_index_records:
            reasons.append("insufficient_index_records")
        if len(profiles) < self.activation_policy.min_distinct_profiles:
            reasons.append("insufficient_distinct_profiles")
        if len(actions) < required_actions:
            reasons.append("insufficient_distinct_actions")
        return MemoryActivationStatus(
            active=not reasons,
            validated_record_count=len(validated),
            distinct_profile_count=len(profiles),
            distinct_action_count=len(actions),
            reasons=tuple(reasons),
        )


class MemoryAwareDecisionProxy:
    """Intercept SemanticDecisionAgent calls without touching SQL generation."""

    def __init__(
        self,
        *,
        delegate: Any,
        controller: StateMemoryController,
        state: Any,
        phase_id: str,
        skill: Any,
    ):
        self.delegate = delegate
        self.controller = controller
        self.state = state
        self.phase_id = phase_id
        self.skill = skill
        self.invoked = False
        self.lookup_result: MemoryLookup | None = None
        self.memory_decision: dict[str, Any] | None = None
        self.token_accounting: dict[str, Any] = {}

    @property
    def memory_hit(self) -> bool:
        return bool(self.lookup_result and self.lookup_result.hit)

    def select_action(self, **kwargs: Any) -> Any:
        self.invoked = True
        skill_input = kwargs.get("skill_input") or {}
        estimated_input_tokens = self._estimate_input_tokens(skill_input)
        action_space = skill_input.get("action_space") or kwargs.get(
            "action_space", []
        )
        self.lookup_result = self.controller.lookup(
            state=self.state,
            phase_id=self.phase_id,
            skill=self.skill,
            action_space=action_space,
        )
        match = self.lookup_result.selected_match
        if match is None:
            before_usage = self._usage_snapshot()
            try:
                return self.delegate.select_action(**kwargs)
            finally:
                reported_tokens = self._reported_prompt_token_delta(
                    before_usage,
                    self._usage_snapshot(),
                )
                self._record_token_accounting(
                    hit=False,
                    estimated_input_tokens=estimated_input_tokens,
                    reported_actual_input_tokens=reported_tokens,
                )
        self.controller.record_reuse_attempt(
            match,
            current_profile=self.lookup_result.profile,
        )

        selected_action = _selected_action(
            match.record.transition.selected_action
        )
        if selected_action is None:
            raise ValueError("Memory transition has no identifiable selected action")
        self.memory_decision = {
            "selected_action": selected_action,
            "confidence": match.reliability,
            "reason": (
                "Reused a validated state-memory decision with profile "
                f"similarity {match.profile_similarity:.3f}."
            ),
        }
        decisions = getattr(self.delegate, "last_decisions", None)
        if isinstance(decisions, dict):
            decisions[self.skill.action_id] = deepcopy(self.memory_decision)
        self._record_token_accounting(
            hit=True,
            estimated_input_tokens=estimated_input_tokens,
            reported_actual_input_tokens=None,
        )
        return selected_action

    def _estimate_input_tokens(self, skill_input: dict[str, Any]) -> int:
        estimator = getattr(self.delegate, "estimate_input_tokens", None)
        if callable(estimator):
            try:
                return max(0, int(estimator(skill_input)))
            except Exception:
                pass
        return estimate_json_tokens(skill_input)

    def _usage_snapshot(self) -> dict[str, int] | None:
        base_agent = getattr(self.delegate, "base_agent", None)
        snapshot = getattr(base_agent, "usage_snapshot", None)
        if not callable(snapshot):
            return None
        try:
            value = snapshot()
        except Exception:
            return None
        return dict(value) if isinstance(value, dict) else None

    @staticmethod
    def _reported_prompt_token_delta(
        before: dict[str, int] | None,
        after: dict[str, int] | None,
    ) -> int | None:
        if before is None or after is None:
            return None
        response_delta = int(after.get("responses_with_prompt_tokens", 0)) - int(
            before.get("responses_with_prompt_tokens", 0)
        )
        if response_delta <= 0:
            return None
        return max(
            0,
            int(after.get("prompt_tokens", 0))
            - int(before.get("prompt_tokens", 0)),
        )

    def _record_token_accounting(
        self,
        *,
        hit: bool,
        estimated_input_tokens: int,
        reported_actual_input_tokens: int | None,
    ) -> None:
        event = {
            "phase_id": self.phase_id,
            "skill_id": self.skill.skill_id,
            "action_id": self.skill.action_id,
            "hit": bool(hit),
            "warm": bool(
                self.lookup_result and self.lookup_result.activation.active
            ),
            "estimated_input_tokens": max(0, int(estimated_input_tokens)),
            "reported_actual_input_tokens": reported_actual_input_tokens,
        }
        self.token_accounting = deepcopy(event)
        self.controller.record_token_accounting(event)


def _selected_action(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ("selected_action", "decision", "action"):
            selected = value.get(key)
            if isinstance(selected, str) and selected.strip():
                return selected.strip()
    return None


def _temporal_failure_signature(
    reuse_trace: MemoryReuseTrace,
) -> dict[str, Any]:
    """Build a conservative temporal exclusion for one failed reuse.

    Refinement is applied only when at least one active temporal dimension
    differs from the validated prototype. Exact-prototype failures continue to
    affect reliability through Nf but do not erase their prior valid scope.
    """

    current = reuse_trace.current_temporal_profile.to_matching_dict()
    remembered = reuse_trace.remembered_temporal_profile.to_matching_dict()
    fields = reuse_trace.active_temporal_fields
    if not fields:
        return {}

    signature: dict[str, Any] = {}
    differs = False
    for path in fields:
        relative_path = path.removeprefix("temporal.")
        current_value = _nested_value(current, relative_path)
        if _missing_temporal_value(current_value):
            continue
        remembered_value = _nested_value(remembered, relative_path)
        signature[path] = deepcopy(current_value)
        if _missing_temporal_value(remembered_value) or canonical_json(
            current_value
        ) != canonical_json(remembered_value):
            differs = True
    return signature if signature and differs else {}


def _nested_value(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _missing_temporal_value(value: Any) -> bool:
    return value in {None, "", UNSET, UNKNOWN}
