"""Distribution-guided priors for semantic action decisions.

This module implements training-free decision learning. It tracks the complete
accepted-data action distribution of each semantic Skill, measures its deficit
from a target distribution, and turns that deficit into a soft prior over the
actions that are valid in the current state.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from threading import RLock
from typing import Any, Iterable
from uuid import uuid4

from .config import get_default_decision_learning_path
from .memory import MemoryValidationResult


ACTION_PRIOR_INSTRUCTION = (
    "The action distribution reflects the current generation imbalance. "
    "Treat semantic validity and the original structural constraints as the "
    "primary criteria. When multiple actions are semantically valid, prefer "
    "actions with higher prior probabilities to improve balanced and "
    "comprehensive dataset coverage. Never choose a higher-priority action if "
    "it makes the decision semantically inappropriate or structurally invalid."
)


# Decisions that materially shape query semantics or function coverage. Skills
# that only instantiate parameters/SQL or choose aliases are excluded.
DEFAULT_DECISION_LEARNING_SKILLS = frozenset(
    {
        # P0: global intent and its selected subtype.
        "TaskFamilySkill",
        "RetrievalSubtypeSkill",
        "ComputationSubtypeSkill",
        "AggregationSubtypeSkill",
        "MaterializationSubtypeSkill",
        # P1: scope semantics, excluding concrete SQL/schema generation.
        "ScopePatternSkill",
        "DeviceScopeSkill",
        "MeasurementScopeSkill",
        "TagScopeSkill",
        # P2-P3: temporal and grouping semantics.
        "TemporalModeSkill",
        "TimeScopeSkill",
        "WindowSizeSkill",
        "WindowStepSkill",
        "GroupParadigmSkill",
        "SegmentAxisSkill",
        "GroupAxisSkill",
        # P4: regime plus dynamic function-family/function coverage.
        "OperatorRegimeSkill",
        "SeriesExpressionSkill",
        "AggregateExpressionSkill",
        "AggregateFunctionFamilySkill",
        "TimeSeriesFunctionFamilySkill",
        "ComponentInstantiationSkill",
        # P6-P9: optional clause semantics and result organization.
        "PredicateModeSkill",
        "FillModeSkill",
        "ResultAlignmentSkill",
        "WritebackModeSkill",
        "OrderingModeSkill",
        "PaginationPolicySkill",
    }
)


# These P4 Skills construct a type-filtered action space for the expression
# currently being processed. Their target distribution is therefore defined on
# that invocation's actual candidates, not on the historical union of every
# function or family ever observed by the Skill.
DYNAMIC_ACTION_SPACE_SKILLS = frozenset(
    {
        "AggregateFunctionFamilySkill",
        "TimeSeriesFunctionFamilySkill",
        "ComponentInstantiationSkill",
    }
)


@dataclass(frozen=True)
class DecisionLearningConfig:
    """Configuration for distribution-guided action-prior computation.

    ``target_distributions`` is optional. Each configured Skill must provide a
    positive weight for every action in its complete action space; weights are
    normalized internally. Unconfigured Skills use a uniform target.
    """

    temperature: float = 1.0
    smoothing: float = 1.0
    activation_sample_threshold: int = 0
    enabled_skill_ids: frozenset[str] = DEFAULT_DECISION_LEARNING_SKILLS
    dynamic_action_space_skill_ids: frozenset[str] = DYNAMIC_ACTION_SPACE_SKILLS
    target_distributions: dict[str, dict[str, float]] = field(default_factory=dict)
    bypass_state_memory: bool = False

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than 0")
        if self.smoothing <= 0:
            raise ValueError("smoothing must be greater than 0")
        if (
            isinstance(self.activation_sample_threshold, bool)
            or not isinstance(self.activation_sample_threshold, int)
            or self.activation_sample_threshold < 0
        ):
            raise ValueError("activation_sample_threshold must be a non-negative integer")
        for skill_id, distribution in self.target_distributions.items():
            if not str(skill_id).strip() or not isinstance(distribution, dict):
                raise ValueError(
                    "target_distributions must map skill ids to action-weight objects"
                )
            if not distribution:
                raise ValueError("configured target distributions cannot be empty")
            for action, weight in distribution.items():
                if not str(action).strip():
                    raise ValueError("target-distribution actions must be non-empty")
                if (
                    isinstance(weight, bool)
                    or not isinstance(weight, (int, float))
                    or not math.isfinite(float(weight))
                    or float(weight) <= 0
                ):
                    raise ValueError(
                        "target-distribution weights must be finite and positive"
                    )

    def target_for(
        self,
        skill_id: str,
        actions: Iterable[str],
    ) -> dict[str, float]:
        normalized_actions = tuple(dict.fromkeys(str(action) for action in actions))
        if not normalized_actions:
            return {}
        configured = self.target_distributions.get(skill_id)
        if configured is None:
            probability = 1.0 / len(normalized_actions)
            return {action: probability for action in normalized_actions}
        missing = [action for action in normalized_actions if action not in configured]
        if missing:
            raise ValueError(
                f"Target distribution for {skill_id} is missing actions: {missing}"
            )
        # Extra configured actions are allowed because dynamic P4 Skills expose
        # only the compatible family/function subset in each invocation.
        denominator = sum(float(configured[action]) for action in normalized_actions)
        return {
            action: float(configured[action]) / denominator
            for action in normalized_actions
        }


@dataclass(frozen=True)
class ActionPriorGuidance:
    """Auditable target-distribution prior and its intermediate values."""

    skill_id: str
    full_action_space: tuple[str, ...]
    valid_actions: tuple[str, ...]
    action_counts: dict[str, int]
    empirical_distribution: dict[str, float]
    target_distribution: dict[str, float]
    distribution_deficit: dict[str, float]
    action_prior: dict[str, float]
    preferred_actions: tuple[str, ...]

    def to_model_fields(self) -> dict[str, Any]:
        """Return only the compact fields that should enter the model prompt."""

        return {
            "action_prior": deepcopy(self.action_prior),
            "preferred_actions": list(self.preferred_actions),
            "action_prior_instruction": ACTION_PRIOR_INSTRUCTION,
        }

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "full_action_space": list(self.full_action_space),
            "valid_actions": list(self.valid_actions),
            "action_counts": deepcopy(self.action_counts),
            "empirical_distribution": deepcopy(self.empirical_distribution),
            "target_distribution": deepcopy(self.target_distribution),
            "distribution_deficit": deepcopy(self.distribution_deficit),
            "action_prior": deepcopy(self.action_prior),
            "preferred_actions": list(self.preferred_actions),
            "instruction": ACTION_PRIOR_INSTRUCTION,
        }


class DecisionLearningStore:
    """Local knowledge base for distribution statistics and trajectories."""

    VERSION = 4
    ALGORITHM = "distribution_aware_online_optimization"

    def __init__(
        self,
        action_counts: dict[str, dict[str, int]] | None = None,
        *,
        action_catalog: dict[str, Iterable[str]] | None = None,
        trajectories: Iterable[dict[str, Any]] | None = None,
        dataset_statistics: dict[str, Any] | None = None,
        storage_path: str | Path | None = None,
    ):
        self.storage_path = Path(
            storage_path or get_default_decision_learning_path()
        )
        self._lock = RLock()
        self._action_counts: dict[str, dict[str, int]] = {}
        self._action_catalog: dict[str, list[str]] = {}
        self._trajectories: list[dict[str, Any]] = []
        self._trajectory_ids: set[str] = set()
        self._dataset_statistics = deepcopy(dataset_statistics or {})

        for skill_id, actions in (action_catalog or {}).items():
            self.register_actions(skill_id, actions)
        for skill_id, counts in (action_counts or {}).items():
            for action, count in counts.items():
                self._set_count(skill_id, action, count)
                self.register_actions(skill_id, [action])
        for record in trajectories or []:
            self.add_trajectory_record(record)

    @property
    def trajectory_count(self) -> int:
        with self._lock:
            return len(self._trajectories)

    @property
    def accepted_trajectory_count(self) -> int:
        with self._lock:
            return sum(bool(item.get("accepted")) for item in self._trajectories)

    @property
    def validated_sample_count(self) -> int:
        """Number of execution-validated NL2SQL samples seen by learning."""

        with self._lock:
            configured_count = self._dataset_statistics.get(
                "validated_sample_count"
            )
            if (
                isinstance(configured_count, int)
                and not isinstance(configured_count, bool)
                and configured_count >= 0
            ):
                return configured_count

            sample_count = 0
            has_sample_counts = False
            for trajectory in self._trajectories:
                if not trajectory.get("accepted"):
                    continue
                trajectory_count = trajectory.get("accepted_sample_count")
                if trajectory_count is None:
                    context = trajectory.get("trajectory_context", {})
                    if isinstance(context, dict):
                        trajectory_count = context.get("sample_count")
                if (
                    isinstance(trajectory_count, int)
                    and not isinstance(trajectory_count, bool)
                    and trajectory_count >= 0
                ):
                    sample_count += trajectory_count
                    has_sample_counts = True
            if has_sample_counts:
                return sample_count
            # Legacy stores counted one accepted trajectory as one sample.
            return sum(bool(item.get("accepted")) for item in self._trajectories)

    @property
    def dataset_statistics(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._dataset_statistics)

    @property
    def trajectories(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._trajectories)

    def action_catalog_for(self, skill_id: str) -> list[str]:
        with self._lock:
            return list(self._action_catalog.get(skill_id, []))

    def register_actions(self, skill_id: str, actions: Iterable[str]) -> None:
        normalized_skill = str(skill_id).strip()
        if not normalized_skill:
            return
        with self._lock:
            catalog = self._action_catalog.setdefault(normalized_skill, [])
            for action in actions:
                normalized_action = str(action).strip()
                if normalized_action and normalized_action not in catalog:
                    catalog.append(normalized_action)

    def count(self, skill_id: str, action: str) -> int:
        with self._lock:
            return self._action_counts.get(skill_id, {}).get(action, 0)

    def counts_for(
        self,
        skill_id: str,
        actions: Iterable[str] | None = None,
    ) -> dict[str, int]:
        with self._lock:
            stored = self._action_counts.get(skill_id, {})
            if actions is None:
                return deepcopy(stored)
            return {str(action): stored.get(str(action), 0) for action in actions}

    def increment_many(
        self,
        decisions: Iterable[tuple[str, str]],
        *,
        amount: int = 1,
    ) -> int:
        """Increment decisions by validated sample count and return total credit."""

        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError("amount must be a non-negative integer")

        credited = 0
        with self._lock:
            for skill_id, action in decisions:
                normalized_skill = str(skill_id).strip()
                normalized_action = str(action).strip()
                if not normalized_skill or not normalized_action:
                    continue
                self.register_actions(normalized_skill, [normalized_action])
                counts = self._action_counts.setdefault(normalized_skill, {})
                counts[normalized_action] = (
                    counts.get(normalized_action, 0) + amount
                )
                credited += amount
        return credited

    def has_trajectory(self, trajectory_id: str) -> bool:
        with self._lock:
            return str(trajectory_id) in self._trajectory_ids

    def add_trajectory_record(self, record: dict[str, Any]) -> bool:
        if not isinstance(record, dict):
            raise TypeError("trajectory record must be a JSON object")
        if not record.get("accepted"):
            return False
        trajectory_id = str(record.get("trajectory_id") or "").strip()
        if not trajectory_id:
            raise ValueError("trajectory record requires trajectory_id")
        with self._lock:
            if trajectory_id in self._trajectory_ids:
                return False
            self._trajectories.append(deepcopy(record))
            self._trajectory_ids.add(trajectory_id)
            return True

    def set_dataset_statistics(self, statistics: dict[str, Any]) -> None:
        if not isinstance(statistics, dict):
            raise TypeError("dataset statistics must be a JSON object")
        with self._lock:
            self._dataset_statistics = deepcopy(statistics)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": self.VERSION,
                "algorithm": self.ALGORITHM,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "action_catalog": deepcopy(self._action_catalog),
                "action_counts": deepcopy(self._action_counts),
                "dataset_statistics": deepcopy(self._dataset_statistics),
                "trajectories": deepcopy(self._trajectories),
            }

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        *,
        storage_path: str | Path | None = None,
    ) -> "DecisionLearningStore":
        if not isinstance(value, dict):
            raise TypeError("decision-learning store must be a JSON object")
        version = value.get("version", 1)
        if version not in {1, 2, 3, cls.VERSION}:
            raise ValueError(f"Unsupported decision-learning store version: {version}")
        counts = value.get("action_counts", {})
        if not isinstance(counts, dict):
            raise TypeError("action_counts must be a JSON object")
        if version == 1:
            return cls(counts, storage_path=storage_path)
        catalog = value.get("action_catalog", {})
        trajectories = value.get("trajectories", [])
        statistics = value.get("dataset_statistics", {})
        if not isinstance(catalog, dict):
            raise TypeError("action_catalog must be a JSON object")
        if not isinstance(trajectories, list):
            raise TypeError("trajectories must be a JSON array")
        if not isinstance(statistics, dict):
            raise TypeError("dataset_statistics must be a JSON object")
        return cls(
            counts,
            action_catalog=catalog,
            trajectories=trajectories,
            dataset_statistics=statistics,
            storage_path=storage_path,
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "DecisionLearningStore":
        source = Path(path) if path is not None else get_default_decision_learning_path()
        if not source.exists():
            return cls(storage_path=source)
        value = json.loads(source.read_text(encoding="utf-8"))
        return cls.from_dict(value, storage_path=source)

    def save(self, path: str | Path | None = None) -> Path:
        """Atomically persist the complete learning knowledge base."""

        destination = Path(path) if path is not None else self.storage_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, destination)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        self.storage_path = destination
        return destination

    def _set_count(self, skill_id: Any, action: Any, count: Any) -> None:
        normalized_skill = str(skill_id).strip()
        normalized_action = str(action).strip()
        if not normalized_skill or not normalized_action:
            raise ValueError("skill and action names must be non-empty")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("action counts must be non-negative integers")
        self._action_counts.setdefault(normalized_skill, {})[
            normalized_action
        ] = count


@dataclass(frozen=True)
class PendingDecision:
    skill_id: str
    action_id: str
    selected_action: str
    phase_id: str
    run_id: str | None = None
    guidance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "action_id": self.action_id,
            "selected_action": self.selected_action,
            "phase_id": self.phase_id,
            "run_id": self.run_id,
            "guidance": deepcopy(self.guidance),
        }


class DecisionLearningController:
    """Compute priors, stage decisions, and credit only accepted trajectories."""

    def __init__(
        self,
        *,
        store: DecisionLearningStore | None = None,
        config: DecisionLearningConfig | None = None,
        autosave: bool = False,
    ):
        self.store = store or DecisionLearningStore()
        self.config = config or DecisionLearningConfig()
        self.autosave = autosave
        self._pending: list[PendingDecision] = []
        self.last_validation_result = MemoryValidationResult.not_run(
            "No decision-learning trajectory has been finalized yet."
        )
        self.last_credited_decision_count = 0

    @property
    def pending_decisions(self) -> tuple[PendingDecision, ...]:
        return tuple(self._pending)

    @property
    def validated_sample_count(self) -> int:
        return self.store.validated_sample_count

    @property
    def prior_active(self) -> bool:
        """Activate only after the configured number of valid samples."""

        return (
            self.validated_sample_count
            > self.config.activation_sample_threshold
        )

    def begin_trajectory(self) -> None:
        self._pending.clear()
        self.last_credited_decision_count = 0
        self.last_validation_result = MemoryValidationResult.not_run(
            "Current decision-learning trajectory has not been accepted."
        )

    def tracks(self, skill_or_id: Any) -> bool:
        skill_id = (
            skill_or_id
            if isinstance(skill_or_id, str)
            else getattr(skill_or_id, "skill_id", "")
        )
        return str(skill_id) in self.config.enabled_skill_ids

    def should_bypass_state_memory(self, skill_or_id: Any) -> bool:
        return (
            self.config.bypass_state_memory
            and self.prior_active
            and self.tracks(skill_or_id)
        )

    def observe_action_space(
        self,
        *,
        skill_id: str,
        actions: Iterable[str],
    ) -> None:
        """Register the actions that are eligible for the current decision."""

        if self.tracks(skill_id):
            self.store.register_actions(
                skill_id,
                (
                    action
                    for action in actions

                ),
            )

    def compute_guidance(
        self,
        *,
        skill_id: str,
        valid_actions: Iterable[str],
        full_actions: Iterable[str] | None = None,
        require_activation: bool = True,
    ) -> ActionPriorGuidance | None:
        """Compare the global action distribution with its target.

        Historical probabilities and deficits are computed over the complete
        Skill action space. The current mask affects only which deficits enter
        the final softmax prior.
        """

        if not self.tracks(skill_id):
            return None
        if require_activation and not self.prior_active:
            return None
        current_valid_actions = tuple(
            dict.fromkeys(
                str(action)
                for action in valid_actions

            )
        )
        if not current_valid_actions:
            return None

        if full_actions is not None:
            full_action_space = tuple(
                dict.fromkeys(
                    str(action)
                    for action in full_actions

                )
            )
            missing_valid_actions = [
                action
                for action in current_valid_actions
                if action not in full_action_space
            ]
            if missing_valid_actions:
                raise ValueError(
                    "valid_actions must be a subset of full_actions; missing "
                    f"{missing_valid_actions}"
                )
        else:
            full_action_space = tuple(
                dict.fromkeys(
                    action
                    for action in [
                        *self.store.action_catalog_for(skill_id),
                        *self.store.counts_for(skill_id).keys(),
                        *current_valid_actions,
                    ]

                )
            )
        if not full_action_space:
            return None
        self.store.register_actions(skill_id, full_action_space)
        counts = self.store.counts_for(skill_id, full_action_space)
        denominator = sum(
            counts[action] + self.config.smoothing for action in full_action_space
        )
        empirical = {
            action: (counts[action] + self.config.smoothing) / denominator
            for action in full_action_space
        }
        target = self.config.target_for(skill_id, full_action_space)
        deficit = {
            action: max(target[action] - empirical[action], 0.0) / target[action]
            for action in full_action_space
        }
        prior = self._softmax(deficit, current_valid_actions)
        original_order = {
            action: index for index, action in enumerate(current_valid_actions)
        }
        preferred = tuple(
            sorted(
                current_valid_actions,
                key=lambda action: (
                    -deficit[action],
                    -prior[action],
                    original_order[action],
                ),
            )
        )
        return ActionPriorGuidance(
            skill_id=skill_id,
            full_action_space=full_action_space,
            valid_actions=current_valid_actions,
            action_counts=counts,
            empirical_distribution=empirical,
            target_distribution=target,
            distribution_deficit=deficit,
            action_prior=prior,
            preferred_actions=preferred,
        )

    def stage_decision(
        self,
        *,
        skill_id: str,
        action_id: str,
        selected_action: str | None,
        phase_id: str,
        run_id: str | None = None,
        guidance: dict[str, Any] | None = None,
    ) -> None:
        if not self.tracks(skill_id) or not selected_action:
            return
        self._pending.append(
            PendingDecision(
                skill_id=str(skill_id),
                action_id=str(action_id),
                selected_action=str(selected_action),
                phase_id=str(phase_id),
                run_id=run_id,
                guidance=deepcopy(guidance or {}),
            )
        )

    def finish_trajectory(
        self,
        *,
        validation_result: MemoryValidationResult,
        state: Any | None = None,
    ) -> int:
        """Persist one trajectory; credit actions only after acceptance."""

        self.last_validation_result = validation_result
        trajectory_id = self._trajectory_id(state)
        if self.store.has_trajectory(trajectory_id):
            self.last_credited_decision_count = 0
            self._pending.clear()
            return 0

        accepted = bool(validation_result.validated and validation_result.passed)
        accepted_sample_count = self._accepted_sample_count(
            validation_result=validation_result,
            state=state,
        ) if accepted else 0
        if not accepted:
            self.last_credited_decision_count = 0
            self._pending.clear()
            return 0
        credited = 0
        credited = self.store.increment_many(
            (
                (decision.skill_id, decision.selected_action)
                for decision in self._pending
            ),
            amount=accepted_sample_count,
        )

        validated_sample_count = (
            self.store.validated_sample_count + accepted_sample_count
        )

        statistics = self._build_dataset_statistics(
            finalized_trajectory_count=self.store.trajectory_count + 1,
            accepted_trajectory_count=(
                self.store.accepted_trajectory_count + 1
            ),
            validated_sample_count=validated_sample_count,
        )
        record = {
            "trajectory_id": trajectory_id,
            "run_id": getattr(state, "run_id", None),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "accepted": True,
            "accepted_sample_count": accepted_sample_count,
            "validation": validation_result.to_dict(),
            "credited_decision_count": credited,
            "decisions": [decision.to_dict() for decision in self._pending],
            "trajectory_context": self._trajectory_context(
                state=state,
                accepted=True,
            ),
            "dataset_statistics_after": deepcopy(statistics),
        }
        self.store.add_trajectory_record(record)
        self.store.set_dataset_statistics(statistics)
        if self.autosave:
            self.store.save()
        self.last_credited_decision_count = credited
        self._pending.clear()
        return credited

    def discard_trajectory(
        self,
        *,
        validation_result: MemoryValidationResult,
    ) -> None:
        """Drop provisional decisions without persisting a trajectory record."""

        self.last_validation_result = validation_result
        self.last_credited_decision_count = 0
        self._pending.clear()

    def _build_dataset_statistics(
        self,
        *,
        finalized_trajectory_count: int,
        accepted_trajectory_count: int,
        validated_sample_count: int,
    ) -> dict[str, Any]:
        skills: dict[str, Any] = {}
        for skill_id in sorted(self.config.enabled_skill_ids):
            actions = [
                action
                for action in self.store.action_catalog_for(skill_id)

            ]
            if not actions:
                continue
            guidance = self.compute_guidance(
                skill_id=skill_id,
                valid_actions=actions,
                require_activation=False,
            )
            if guidance is None:
                continue
            actions = list(guidance.full_action_space)
            counts = guidance.action_counts
            action_count = len(actions)
            distribution = guidance.empirical_distribution
            target_distribution = guidance.target_distribution
            normalized_entropy = self._normalized_entropy(
                distribution,
                action_count,
            )
            covered_action_count = sum(counts[action] > 0 for action in actions)
            total_variation_distance = 0.5 * sum(
                abs(distribution[action] - target_distribution[action])
                for action in actions
            )
            skills[skill_id] = {
                "action_catalog": list(actions),
                "action_counts": deepcopy(counts),
                "empirical_distribution": deepcopy(distribution),
                "target_distribution": deepcopy(target_distribution),
                "distribution_deficit": deepcopy(
                    guidance.distribution_deficit
                ),
                "covered_action_count": covered_action_count,
                "action_count": action_count,
                "coverage_rate": covered_action_count / action_count,
                "mean_distribution_deficit": (
                    sum(guidance.distribution_deficit.values()) / action_count
                ),
                "total_variation_distance": total_variation_distance,
                "normalized_entropy": normalized_entropy,
                "next_action_prior": deepcopy(guidance.action_prior),
                "preferred_actions": list(guidance.preferred_actions),
            }
        return {
            "algorithm": DecisionLearningStore.ALGORITHM,
            "finalized_trajectory_count": finalized_trajectory_count,
            "accepted_trajectory_count": accepted_trajectory_count,
            "validated_sample_count": validated_sample_count,
            "activation_sample_threshold": (
                self.config.activation_sample_threshold
            ),
            "action_prior_active": (
                validated_sample_count
                > self.config.activation_sample_threshold
            ),
            "rejected_or_unvalidated_trajectory_count": (
                finalized_trajectory_count - accepted_trajectory_count
            ),
            "total_credited_decision_count": sum(
                sum(skill_statistics["action_counts"].values())
                for skill_statistics in skills.values()
            ),
            "skills": skills,
        }

    @staticmethod
    def _accepted_sample_count(
        *,
        validation_result: MemoryValidationResult,
        state: Any | None,
    ) -> int:
        """Resolve how many validated samples share the staged decisions."""

        reported_count = validation_result.details.get("accepted_count")
        if (
            isinstance(reported_count, int)
            and not isinstance(reported_count, bool)
            and reported_count > 0
        ):
            return reported_count
        state_samples = getattr(state, "final_samples", None)
        if isinstance(state_samples, list) and state_samples:
            return len(state_samples)
        # A custom validator may report only a boolean success. Preserve the
        # historical one-successful-trajectory/one-sample interpretation.
        return 1

    def _trajectory_id(self, state: Any | None) -> str:
        run_id = str(getattr(state, "run_id", "") or "").strip()
        if run_id:
            return run_id
        for decision in self._pending:
            if decision.run_id:
                return str(decision.run_id)
        return f"decision-learning-{uuid4().hex}"

    @staticmethod
    def _trajectory_context(
        *,
        state: Any | None,
        accepted: bool,
    ) -> dict[str, Any]:
        if state is None:
            return {"samples": []}
        input_request = deepcopy(getattr(state, "input_request", {}) or {})
        query_frame = getattr(state, "query_frame", None)
        if query_frame is not None and hasattr(query_frame, "to_flat_dict"):
            query_frame_value = query_frame.to_flat_dict()
        else:
            query_frame_value = {}
        samples = (
            deepcopy(getattr(state, "final_samples", []) or [])
            if accepted
            else []
        )
        return {
            "database": input_request.get("database"),
            "generation_request": input_request.get("generation_request"),
            "user_generation_goal": input_request.get("user_generation_goal"),
            "phase_progress": deepcopy(getattr(state, "phase_progress", []) or []),
            "query_frame": deepcopy(query_frame_value),
            "natural_language_semantics": deepcopy(
                getattr(state, "natural_language_semantics", {}) or {}
            ),
            "sample_count": len(samples),
            "samples": samples,
        }

    @staticmethod
    def _normalized_entropy(
        distribution: dict[str, float],
        action_count: int,
    ) -> float:
        if action_count <= 1:
            return 1.0
        entropy = -sum(
            probability * math.log(probability)
            for probability in distribution.values()
            if probability > 0
        )
        return entropy / math.log(action_count)

    def _softmax(
        self,
        scores: dict[str, float],
        actions: tuple[str, ...],
    ) -> dict[str, float]:
        scaled = {
            action: scores[action] / self.config.temperature for action in actions
        }
        maximum = max(scaled.values())
        exponentials = {
            action: math.exp(scaled[action] - maximum) for action in actions
        }
        denominator = sum(exponentials.values())
        return {
            action: exponentials[action] / denominator for action in actions
        }


class DecisionLearningAgentProxy:
    """Inject distribution-guided priors into SemanticDecisionAgent calls."""

    def __init__(
        self,
        *,
        delegate: Any,
        controller: DecisionLearningController,
        skill_id: str,
    ):
        self.delegate = delegate
        self.controller = controller
        self.skill_id = skill_id
        self.invoked = False
        self.last_guidance: ActionPriorGuidance | None = None

    def _inject_guidance(
        self,
        skill_input: dict[str, Any],
    ) -> dict[str, Any]:
        prepared_input = deepcopy(skill_input)
        action_space = list(prepared_input.get("action_space") or [])
        guidance = self.controller.compute_guidance(
            skill_id=self.skill_id,
            valid_actions=action_space,
            # Normalize the prior over grammar-compatible candidates.
            full_actions=action_space,
        )
        self.last_guidance = guidance
        if guidance is not None:
            prepared_input.update(guidance.to_model_fields())
        return prepared_input

    def select_action(self, **kwargs: Any) -> Any:
        original_skill_input = kwargs.get("skill_input")
        if isinstance(original_skill_input, dict):
            skill_input = deepcopy(original_skill_input)
        else:
            action_id = str(kwargs.get("action_id") or "")
            state = deepcopy(kwargs.get("state") or {})
            skill_input = {
                "phase_id": action_id.split("A", 1)[0],
                "skill_id": kwargs.get("skill_id") or self.skill_id,
                "action_id": action_id,
                "decision_name": kwargs.get("decision_name") or "",
                "state": state,
                "context": kwargs.get("context") or "",
                "schema_view": deepcopy(
                    kwargs.get("schema_view") or state.get("schema", {})
                ),
                "query_frame": deepcopy(
                    kwargs.get("query_frame")
                    or state.get("query_frame_flat", {})
                ),
                "action_space": list(kwargs.get("action_space") or []),
                "candidate_descriptions": deepcopy(
                    kwargs.get("candidate_descriptions") or {}
                ),
            }
        skill_input = self._inject_guidance(skill_input)
        kwargs = dict(kwargs)
        kwargs["skill_input"] = skill_input
        self.invoked = True
        return self.delegate.select_action(**kwargs)

    def decide_structured(self, *, skill_input: dict[str, Any], **kwargs: Any) -> Any:
        """Keep action priors active in the P4 exploration loop."""
        prepared_input = self._inject_guidance(skill_input)
        self.invoked = True
        if hasattr(self.delegate, "decide_structured"):
            return self.delegate.decide_structured(skill_input=prepared_input, **kwargs)
        return self.delegate.select_action(
            skill_input=prepared_input, action_id=prepared_input.get("action_id"),
            skill_id=prepared_input.get("skill_id"), decision_name=prepared_input.get("decision_name"),
        )

    def select_action_and_rewrite_expression(
        self,
        *,
        skill_input: dict[str, Any],
        expression: str,
    ) -> Any:
        """Inject priors into the specialized P4 A5/A6 decision API."""

        prepared_input = self._inject_guidance(skill_input)
        self.invoked = True
        return self.delegate.select_action_and_rewrite_expression(
            skill_input=prepared_input,
            expression=expression,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)
